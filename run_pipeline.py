"""
End-to-end CCTV 분석 파이프라인 오케스트레이터

각 단계별 수치 검증 → PASS/FAIL 게이팅 → 다음 단계 진행
기준치 미통과 시 중단 (--force 로 강제 진행 가능)

실행: python run_pipeline.py [--day 10] [--display] [--skip-mosaic]
                              [--skip-validate] [--force]
"""
import argparse, json, sys, time, cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

BASE_DIR    = Path(__file__).parent
MOSAIC_DIR  = BASE_DIR / "mosaic"
MERGED_DIR  = BASE_DIR / "merged"
LOGS_DIR    = BASE_DIR / "logs"
WEIGHTS_DIR = BASE_DIR / "weights"
LOGS_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════════════════════
#  [임시 디버그] ch3 바운딩박스 시각화 영상 저장  ← 추후 제거 예정
# ══════════════════════════════════════════════════════════════════════════════

def _save_ch3_debug_video(day):
    """
    ch3 모자이크 영상에 사람 감지 바운딩박스 + 트랙 ID를 그려 MP4 저장.
    출력: debug_ch3_{day}.mp4
    [임시 기능 - 추후 제거 예정]
    """
    src  = MOSAIC_DIR / f"ch3_{day}.mp4"
    dest = BASE_DIR   / f"debug_ch3_{day}.mp4"

    if not src.exists():
        return False, f"{src.name} 없음"

    from ultralytics import YOLO as _YOLO
    model  = _YOLO("yolov8n-pose.pt")
    cap    = cv2.VideoCapture(str(src))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(dest),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 사람 추적 (track ID 포함)
        res = model.track(frame, persist=True, classes=[0],
                          conf=0.4, device=DEVICE, verbose=False)

        if res[0].boxes is not None and res[0].boxes.id is not None:
            ids  = res[0].boxes.id.cpu().numpy().astype(int)
            xyxy = res[0].boxes.xyxy.cpu().numpy().astype(int)
            conf = res[0].boxes.conf.cpu().numpy()

            for tid, box, c in zip(ids, xyxy, conf):
                x1, y1, x2, y2 = box
                # 바운딩박스
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 라벨: 트랙 ID + 신뢰도
                label = f"ID:{tid}  {c:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame,
                              (x1, y1 - th - 6), (x1 + tw + 4, y1),
                              (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # 좌상단 HUD
        n_persons = len(res[0].boxes) if res[0].boxes is not None else 0
        hud = f"ch3 | frame {fi}/{total} | 감지: {n_persons}명"
        cv2.rectangle(frame, (0, 0), (W, 30), (0, 0, 0), -1)
        cv2.putText(frame, hud, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 1)

        writer.write(frame)
        fi += 1
        if fi % 500 == 0:
            print(f"    debug video: {fi}/{total} ({fi/total*100:.1f}%)")

    cap.release()
    writer.release()
    return True, f"저장: {dest.name}  ({total}프레임)"


# ══════════════════════════════════════════════════════════════════════════════
#  BoT-SORT  &  MiVOLO v2  헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _setup_botsort():
    """
    Ultralytics YOLO.track() 전역에 BoT-SORT 추적기 주입.
    analyze.py 수정 없이 monkey-patch 방식으로 적용.
    반환: (성공 여부, 메시지)
    """
    try:
        from ultralytics import YOLO as _YOLO
        _orig = _YOLO.track

        def _botsort_track(self, source=None, stream=False, **kw):
            kw.setdefault("tracker", "botsort.yaml")   # BoT-SORT 주입
            return _orig(self, source=source, stream=stream, **kw)

        _YOLO.track = _botsort_track
        return True, "BoT-SORT 적용 (Re-ID 내장, 외모 기반 트랙 재연결)"
    except Exception as e:
        return False, f"설정 실패 → ByteTrack 유지 ({e})"


def _run_mivolo_pass(day):
    """
    MiVOLO v2 (바디 전용) 로 visit_data.json의 gender / is_adult 재분류.
    run_analysis() 완료 후 후처리로 실행.

    필요 파일:
      weights/mivolo_d1.pth.tar      ← MiVOLO GitHub Releases
      weights/yolov8x_person_face.pt ← 옵션 (없으면 YOLO 기본 사용)

    반환: (성공 여부, 요약 메시지)
    """
    vd_path = BASE_DIR / "visit_data.json"
    if not vd_path.exists():
        return False, "visit_data.json 없음"

    # ── MiVOLO 임포트 확인 ───────────────────────────────────────────────────
    try:
        from mivolo.predictor import Predictor as _MiVOLO
    except ImportError:
        return False, ("mivolo 미설치 → 아래 명령 실행 후 재시도\n"
                       "  pip install git+https://github.com/WildChlamydia/MiVOLO.git")

    # ── 가중치 확인 ──────────────────────────────────────────────────────────
    mivolo_ckpt   = WEIGHTS_DIR / "mivolo_d1.pth.tar"
    detector_ckpt = WEIGHTS_DIR / "yolov8x_person_face.pt"
    if not mivolo_ckpt.exists():
        return False, (f"가중치 없음: {mivolo_ckpt.name}\n"
                       "  https://github.com/WildChlamydia/MiVOLO/releases 에서 다운로드")

    print("  [MiVOLO] 모델 로딩...")
    try:
        predictor = _MiVOLO(
            detector_weights=str(detector_ckpt) if detector_ckpt.exists() else None,
            checkpoint=str(mivolo_ckpt),
            device=DEVICE,
            with_persons=True,
            disable_faces=True,   # 얼굴 모자이크 환경 → 바디만 사용
            draw=False,
        )
    except Exception as e:
        return False, f"MiVOLO 로드 실패: {e}"

    # ── 날짜 기준선 추출 (timeline.json 첫 클립) ─────────────────────────────
    date_str = None
    for cam in ["ch1", "ch3", "ch6"]:
        tl_path = MERGED_DIR / f"{cam}_{day}_timeline.json"
        if tl_path.exists():
            with open(tl_path, encoding="utf-8") as f:
                tl = json.load(f)
            date_str = tl["clips"][0]["time_start"][:10]   # "2026-04-10"
            break
    if date_str is None:
        return False, "timeline.json 없음 → 날짜 기준 불명"

    # ── visit_data 로드 ───────────────────────────────────────────────────────
    with open(vd_path, encoding="utf-8") as f:
        visits = json.load(f)

    # 방문자별 시간 창 { vid: (entry_dt, exit_dt) }
    def _to_dt(t_str):
        if t_str is None:
            return None
        return datetime.strptime(f"{date_str} {t_str}", "%Y-%m-%d %H:%M:%S")

    visit_windows = {
        vid: (_to_dt(v.get("entry_time")), _to_dt(v.get("exit_time")))
        for vid, v in visits.items()
        if v.get("entry_time") and v.get("exit_time")
    }

    # ── 카메라별 MiVOLO 패스 ─────────────────────────────────────────────────
    from analyze import load_timeline, frame_to_dt

    SAMPLE_SKIP = 20   # 20프레임마다 1회 처리
    gender_votes  = {vid: [] for vid in visit_windows}
    adult_votes   = {vid: [] for vid in visit_windows}

    for cam in ["ch3", "ch6", "ch1"]:
        video_path = MOSAIC_DIR / f"{cam}_{day}.mp4"
        if not video_path.exists():
            continue
        tl = load_timeline(cam, day)
        if tl is None:
            continue

        cap = cv2.VideoCapture(str(video_path))
        fi  = 0
        print(f"  [MiVOLO] {cam} 패스 중...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % SAMPLE_SKIP != 0:
                fi += 1
                continue

            frame_dt = frame_to_dt(fi, tl)
            if frame_dt is None:
                fi += 1
                continue

            # 이 프레임 시각에 해당하는 방문자 목록
            active = [vid for vid, (es, ex) in visit_windows.items()
                      if es <= frame_dt <= ex]
            if not active:
                fi += 1
                continue

            # MiVOLO 실행
            try:
                out, _ = predictor.recognize(frame)
                if out is not None and hasattr(out, "persons"):
                    for person in out.persons:
                        g = ("M" if str(getattr(person, "gender", "")).lower() == "male"
                             else "F" if str(getattr(person, "gender", "")).lower() == "female"
                             else None)
                        age = getattr(person, "age", None)
                        is_adult = (age >= 18) if isinstance(age, (int, float)) else None
                        for vid in active:
                            if g:       gender_votes[vid].append(g)
                            if is_adult is not None:
                                adult_votes[vid].append(is_adult)
            except Exception:
                pass

            fi += 1
        cap.release()

    # ── 다수결로 visit_data 업데이트 ────────────────────────────────────────
    updated = 0
    for vid in visit_windows:
        gv = gender_votes[vid]
        av = adult_votes[vid]
        if not gv:
            continue
        best_g = Counter(gv).most_common(1)[0][0]
        best_a = Counter(av).most_common(1)[0][0] if av else visits[vid].get("is_adult")
        visits[vid]["gender"]   = best_g
        visits[vid]["is_adult"] = best_a
        updated += 1

    with open(vd_path, "w", encoding="utf-8") as f:
        json.dump(visits, f, ensure_ascii=False, indent=2, default=str)

    return True, f"{updated}/{len(visit_windows)}명 gender/age MiVOLO로 재분류 완료"

# ── 게이팅 기준치 ─────────────────────────────────────────────────────────────
GATES = {
    "data":     {"clips_per_cam":   1,    "timeline_ratio": 1.0},
    "mosaic":   {"proc_fps":        3.0},
    "zones":    {"zone_ratio":      0.67},   # 9개 중 6개 이상
    "analyze":  {"tracks_per_cam":  3,    "proc_fps": 1.0},
    "export":   {"visitors":        1},
    "validate": {"score":           0.30},
}

W = 70   # 출력 너비

# ── 출력 포맷 헬퍼 ────────────────────────────────────────────────────────────

def _bar(ch="═"): return ch * W

def stage_header(n, total, title):
    print(f"\n{_bar()}")
    print(f"  [{n}/{total}] {title}")
    print(_bar())

def metric_row(label, value, status, gate=""):
    icon = "✅" if status == "pass" else ("❌" if status == "fail" else "ℹ ")
    g    = f"  (기준: {gate})" if gate else ""
    print(f"  {label:<28}{str(value):<16}{icon}{g}")

def stage_footer(passed, elapsed, n, total):
    print(_bar("─"))
    mark   = "✅" if passed else "❌"
    result = "PASSED" if passed else "FAILED"
    print(f"  {mark} [{n}/{total}] {result}   ({elapsed:.1f}초)")
    print(_bar())


# ── STAGE 1: 데이터 확인 ──────────────────────────────────────────────────────

def stage_data(day, n, total):
    stage_header(n, total, "데이터 확인")
    t0 = time.time(); passed = True

    cams = ["ch1", "ch3", "ch6"]
    gate = GATES["data"]
    all_clips = 0

    # ── merged 영상 확인 ──────────────────────────────────────────────────────
    for cam in cams:
        merged = list(MERGED_DIR.glob(f"{cam}_{day}.mp4"))
        n_clips = len(merged)
        ok = n_clips >= gate["clips_per_cam"]
        if not ok: passed = False
        metric_row(f"merged {cam}_{day}.mp4", f"{n_clips}개",
                   "pass" if ok else "fail", f"≥{gate['clips_per_cam']}")
        all_clips += n_clips

    # ── 타임라인 JSON 확인 → 없으면 자동 생성 ────────────────────────────────
    tl_missing = [c for c in cams
                  if not (MERGED_DIR / f"{c}_{day}_timeline.json").exists()]

    if tl_missing:
        metric_row("타임라인 JSON (생성 전)",
                   f"{len(cams)-len(tl_missing)}/{len(cams)}", "info")
        print(f"  → 누락 카메라: {tl_missing}  자동 생성 시작...")
        sys.path.insert(0, str(BASE_DIR))
        from merge import get_groups, merge_group
        groups = get_groups()
        for cam in tl_missing:
            key = f"{cam}_{day}"
            if key in groups:
                merge_group(key, groups[key])
            else:
                print(f"    [경고] {key} 클립 없음 — organize.py 먼저 실행 필요")

    tl_ok    = sum(1 for c in cams
                   if (MERGED_DIR / f"{c}_{day}_timeline.json").exists())
    tl_ratio = tl_ok / len(cams)
    ok_tl    = tl_ratio >= gate["timeline_ratio"]
    if not ok_tl: passed = False
    metric_row("타임라인 JSON", f"{tl_ok}/{len(cams)}",
               "pass" if ok_tl else "fail", "100%")

    # ── ch1 시간 범위 (기준 창) INFO ──────────────────────────────────────────
    import json as _json
    ch1_tl_path = MERGED_DIR / f"ch1_{day}_timeline.json"
    if ch1_tl_path.exists():
        with open(ch1_tl_path, encoding="utf-8") as f:
            ch1_tl = _json.load(f)
        clips = ch1_tl.get("clips", [])
        if clips:
            ch1_start = clips[0]["time_start"]
            ch1_end   = clips[-1]["time_end"]
            metric_row("ch1 기준 창 (영업 시간)", f"{ch1_start[11:]} ~ {ch1_end[11:]}", "info")

    # ── 총 영상 길이 INFO ─────────────────────────────────────────────────────
    import cv2 as _cv2
    total_min = 0.0
    for mp4 in MERGED_DIR.glob(f"*_{day}.mp4"):
        cap = _cv2.VideoCapture(str(mp4))
        fps = cap.get(_cv2.CAP_PROP_FPS) or 25
        fc  = cap.get(_cv2.CAP_PROP_FRAME_COUNT)
        total_min += fc / fps / 60
        cap.release()
    metric_row("총 영상 길이 (3개 합산)", f"{total_min:.1f}분", "info")

    stage_footer(passed, time.time()-t0, n, total)
    return passed, {"total_merged": all_clips, "timeline_ratio": tl_ratio}


# ── STAGE 2: 얼굴 모자이크 ────────────────────────────────────────────────────

def stage_mosaic(day, n, total):
    stage_header(n, total, "얼굴 모자이크 (프라이버시)")
    t0 = time.time(); passed = True; gate = GATES["mosaic"]

    from face_mosaic import download_model, process_video, DEVICE, USE_FP16
    from ultralytics import YOLO
    import torch

    MOSAIC_DIR.mkdir(exist_ok=True)
    download_model()
    model_path = BASE_DIR / "yolov8n-face.pt"
    model = YOLO(str(model_path))
    if USE_FP16 and DEVICE == "cuda":
        model.model.half()

    videos = sorted(MERGED_DIR.glob(f"*_{day}.mp4"))
    if not videos:
        metric_row("영상 파일", "없음", "fail", "≥1개")
        stage_footer(False, time.time()-t0, n, total)
        return False, {}

    all_m = []
    for v in videos:
        print(f"  처리: {v.name}")
        m = process_video(v, model)
        all_m.append(m)

    for m in all_m:
        if m is None: continue
        if m["skipped"]:
            metric_row(f"처리 속도 (skip)", "이미 존재", "pass", f"≥{gate['proc_fps']}fps")
            continue
        ok = m["fps"] >= gate["proc_fps"]
        if not ok: passed = False
        metric_row(f"처리 속도 ({m.get('cam','?')})", f"{m['fps']}fps",
                   "pass" if ok else "fail", f"≥{gate['proc_fps']}fps")
        metric_row(f"  얼굴 검출 (총)", f"{m['faces']}건", "info")

    stage_footer(passed, time.time()-t0, n, total)
    return passed, {"mosaic_metrics": all_m}


# ── STAGE 3: 구역 설정 (미설정 시 GUI 자동 실행) ─────────────────────────────

def _zone_cam_stats(zones):
    """카메라별 {cam: (done, skipped)} 반환"""
    total_z = len(zones)
    stats = {}
    for cam in ["ch3", "ch6", "ch1"]:
        done    = sum(1 for z in zones
                      if isinstance(z.get("polygons",{}).get(cam,[]), list)
                      and len(z.get("polygons",{}).get(cam,[])) >= 3)
        skipped = sum(1 for z in zones
                      if z.get("polygons",{}).get(cam) == "skip")
        stats[cam] = (done, skipped, total_z - done - skipped)
    return stats


def stage_zones(day, n, total):
    stage_header(n, total, "구역 설정")
    t0 = time.time(); passed = True; gate = GATES["zones"]

    import json as _json
    import importlib

    zones_file = BASE_DIR / "zones.json"
    if not zones_file.exists():
        metric_row("zones.json", "없음", "fail")
        stage_footer(False, time.time()-t0, n, total)
        return False, {}

    with open(zones_file, encoding="utf-8") as f:
        zones = _json.load(f)
    total_z = len(zones)

    # ── 미설정 카메라 감지 → GUI 자동 실행 ───────────────────────────────────
    stats = _zone_cam_stats(zones)
    needs_config = [
        cam for cam in ["ch3", "ch6", "ch1"]
        if stats[cam][0] / total_z < gate["zone_ratio"]   # done 비율 미달
        and stats[cam][2] > 0                             # 미설정이 남아있음
    ]

    if needs_config:
        metric_row("미설정 카메라", str(needs_config), "info")
        print(f"\n  → 구역 설정 GUI 자동 실행: {needs_config}\n")

        spec = importlib.util.spec_from_file_location(
            "zone_config", BASE_DIR / "zone_config.py")
        zc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(zc)

        zones_data = zc.load_zones()
        for cam in needs_config:
            zones_data = zc.configure_camera(cam, day, zones_data)
            zc.save_zones(zones_data)
            print(f"  → {cam} 저장 완료")

        # 저장 후 재로드
        with open(zones_file, encoding="utf-8") as f:
            zones = _json.load(f)
        stats = _zone_cam_stats(zones)

    # ── 최종 현황 출력 ────────────────────────────────────────────────────────
    for cam, (done, skip, empty) in stats.items():
        metric_row(f"{cam} 구역",
                   f"설정={done}  skip={skip}  미설정={empty}",
                   "info")

    # 게이트: ch3 or ch6 중 하나가 기준 이상이면 통과
    best_ratio = max(
        stats["ch3"][0] / total_z if total_z else 0,
        stats["ch6"][0] / total_z if total_z else 0,
    )
    ok = best_ratio >= gate["zone_ratio"]
    if not ok: passed = False
    metric_row("최고 설정률 (ch3 or ch6)",
               f"{best_ratio*100:.0f}%",
               "pass" if ok else "fail",
               f"≥{gate['zone_ratio']*100:.0f}%")

    # 스캐너 구역 설정 여부 (구매 감지에 필요)
    scanner_zone = next((z for z in zones if z.get("id") == "scanner"), None)
    if scanner_zone:
        pts = scanner_zone.get("polygons", {}).get("ch1", [])
        scanner_set = isinstance(pts, list) and len(pts) >= 3
        metric_row("스캐너 구역 (ch1)",
                   "설정됨 ✅" if scanner_set else "미설정 → 상품 감지 비활성",
                   "pass" if scanner_set else "info")

    stage_footer(passed, time.time()-t0, n, total)
    return passed, {"cam_stats": {c: v[0] for c, v in stats.items()}, "total": total_z}


# ── STAGE 4: 영상 분석 ────────────────────────────────────────────────────────

def stage_analyze(day, display, n, total):
    stage_header(n, total, "영상 분석 (추적 + Re-ID)")
    t0 = time.time(); passed = True; gate = GATES["analyze"]

    # ── BoT-SORT 적용 ─────────────────────────────────────────────────────────
    botsort_ok, botsort_msg = _setup_botsort()
    metric_row("추적기", botsort_msg, "pass" if botsort_ok else "info")

    # ── 상품 DB 확인 ──────────────────────────────────────────────────────────
    db_file = BASE_DIR / "product_db" / "embeddings.npz"
    if db_file.exists():
        _db = np.load(db_file, allow_pickle=True)
        n_products = len(_db["names"])
        metric_row("상품 DB", f"{n_products}개 상품 로드됨", "pass" if n_products > 0 else "fail")
    else:
        metric_row("상품 DB", "없음 → 상품 분류 비활성  (build_product_db.py 실행 필요)", "info")

    from analyze import run_analysis
    visits, summary = run_analysis(day=day, display=display)

    if visits is None:
        metric_row("분석 결과", summary.get("error","실패"), "fail")
        stage_footer(False, time.time()-t0, n, total)
        return False, {}

    # 카메라별 추적 수 / 처리 속도
    for m in summary.get("per_cam", []):
        ok_t = m["tracks"] >= gate["tracks_per_cam"]
        ok_f = m["proc_fps"] >= gate["proc_fps"]
        if not ok_t or not ok_f: passed = False
        metric_row(f"{m['cam']} 추적 수",
                   f"{m['tracks']}명",
                   "pass" if ok_t else "fail",
                   f"≥{gate['tracks_per_cam']}")
        metric_row(f"{m['cam']} 처리 속도",
                   f"{m['proc_fps']}fps",
                   "pass" if ok_f else "fail",
                   f"≥{gate['proc_fps']}fps")
        metric_row(f"{m['cam']} 성별 감지율",
                   f"{m['gender_rate']*100:.1f}%", "info")
        metric_row(f"{m['cam']} 성별 신뢰도 avg",
                   f"{m['gender_conf_mean']:.3f}", "info")
        metric_row(f"{m['cam']} 성인 비율",
                   f"{m['adult_rate']*100:.1f}%", "info")

    reid = summary.get("reid", {})
    metric_row("글로벌 방문자 (ch3+ch6)", f"{reid.get('store_visitors',0)}명", "info")
    if "ch1_matched" in reid:
        match_rate = reid["ch1_matched"] / max(reid.get("ch1_tracks",1), 1)
        metric_row("ch1 Re-ID 매칭률",
                   f"{reid['ch1_matched']}/{reid.get('ch1_tracks',0)} ({match_rate*100:.1f}%)",
                   "info")

    # 구매 상품 감지 결과
    n_purchased = summary.get("purchasers", 0)
    n_items_det = sum(
        len(v.get("purchased_items", []))
        for v in (visits or {}).values()
    )
    metric_row("구매자 수",        f"{n_purchased}명",   "info")
    metric_row("상품 감지 건수",   f"{n_items_det}건",   "info")

    # ── MiVOLO v2 후처리 (gender / age 재분류) ───────────────────────────────
    mivolo_ok, mivolo_msg = _run_mivolo_pass(day)
    metric_row("MiVOLO 재분류", mivolo_msg, "pass" if mivolo_ok else "info")

    # ── [임시] ch3 바운딩박스 디버그 영상 저장 ────────────────────────────────
    print(f"\n  [디버그] ch3 바운딩박스 영상 생성 중...")
    dbg_ok, dbg_msg = _save_ch3_debug_video(day)
    metric_row("[임시] ch3 debug MP4", dbg_msg, "pass" if dbg_ok else "info")

    stage_footer(passed, time.time()-t0, n, total)
    return passed, summary


# ── STAGE 5: CSV 출력 ─────────────────────────────────────────────────────────

def stage_export(n, total):
    stage_header(n, total, "CSV 출력")
    t0 = time.time(); passed = True; gate = GATES["export"]

    result_file = BASE_DIR / "visit_data.json"
    if not result_file.exists():
        metric_row("visit_data.json", "없음", "fail")
        stage_footer(False, time.time()-t0, n, total)
        return False, {}

    import importlib, sys as _sys
    # export.py를 모듈로 실행
    spec = importlib.util.spec_from_file_location("export", BASE_DIR/"export.py")
    exp  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)

    visits_raw, zones = exp.load_data()
    df_p = exp.build_purchases(visits_raw, zones)
    df_c = exp.build_class_kpi(visits_raw, zones)
    df_z = exp.build_zone_kpi(visits_raw, zones)
    # ── 구역ID/구역명 정규화 (name→id 치환) ──────────────────────────────────
    # zones.json "name"(한글) → "id"(영문) 매핑 생성
    import json as _json
    with open(BASE_DIR / "zones.json", encoding="utf-8") as _f:
        _zones = _json.load(_f)
    name_to_id   = {z["name"]: z["id"]   for z in _zones}
    name_to_name = {z["name"]: z["name"] for z in _zones}   # 이미 한글, 그대로

    # ice_batch_purchases: 구역ID = 영문 id, 구역명 = 한글 name
    if "구역ID" in df_p.columns:
        df_p["구역ID"] = df_p["구역ID"].map(
            lambda x: name_to_id.get(x, name_to_id.get(x.replace("_"," "), x))
        )
    if "구역명" in df_p.columns:
        df_p["구역명"] = df_p["구역명"].map(
            lambda x: name_to_name.get(x, x)
        )

    # ice_zone_kpi: 구역 컬럼도 동일하게 영문 id로 변환
    if "구역" in df_z.columns:
        df_z["구역"] = df_z["구역"].map(
            lambda x: name_to_id.get(x, name_to_id.get(x.replace("_"," "), x))
        )

    df_p.to_csv(exp.OUT_PURCHASES, index=False, encoding="utf-8-sig")
    df_c.to_csv(exp.OUT_CLASS_KPI, index=False, encoding="utf-8-sig")
    df_z.to_csv(exp.OUT_ZONE_KPI,  index=False, encoding="utf-8-sig")

    n_visitors  = df_p["에이전트ID"].nunique()
    n_purchase  = int((df_p.groupby("에이전트ID")["가격"].sum() > 0).sum())
    conv_rate   = round(n_purchase / n_visitors * 100, 1) if n_visitors > 0 else 0

    ok = n_visitors >= gate["visitors"]
    if not ok: passed = False
    metric_row("총 방문자 수",     f"{n_visitors}명",   "pass" if ok else "fail", f"≥{gate['visitors']}")
    metric_row("구매자 수",        f"{n_purchase}명",   "info")
    metric_row("구매 전환율",      f"{conv_rate}%",     "info")

    # 성별 분포
    if "고객분류명" in df_c.columns:
        for _, row in df_c.iterrows():
            metric_row(f"  {row['분류명']}",
                       f"방문={row['방문객수']}명 구매전환={row['구매전환율(%)']}%", "info")

    metric_row("출력 파일", "ice_batch_purchases.csv", "info")
    metric_row("출력 파일", "ice_class_kpi.csv",       "info")
    metric_row("출력 파일", "ice_zone_kpi.csv",        "info")

    stage_footer(passed, time.time()-t0, n, total)
    return passed, {"visitors": n_visitors, "purchasers": n_purchase,
                    "conv_rate": conv_rate}


# ── STAGE 6: 검증 (GT 있을 때만) ─────────────────────────────────────────────

def stage_validate(n, total):
    stage_header(n, total, "결과 검증 (GT 비교)")
    t0 = time.time(); passed = True; gate = GATES["validate"]

    gt_dir = BASE_DIR / "ground_truth"
    required = ["ice_batch_purchases.csv", "ice_class_kpi.csv", "ice_zone_kpi.csv"]
    if not all((gt_dir / f).exists() for f in required):
        metric_row("GT 파일", "없음 → 검증 건너뜀", "info")
        stage_footer(True, time.time()-t0, n, total)
        return True, {}

    import importlib
    spec = importlib.util.spec_from_file_location("validate", BASE_DIR/"validate.py")
    val  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(val)

    result = val.run_validation()   # validate.py에 run_validation() 함수 필요
    if result is None:
        metric_row("검증 실행", "실패", "fail")
        stage_footer(False, time.time()-t0, n, total)
        return False, {}

    for key, val_num, label in [
        ("cls_acc",     result.get("cls_acc",0),   "분류 정확도"),
        ("purchase_f1", result.get("purchase_f1",0),"구매 F1"),
        ("zone_jaccard",result.get("zone_jaccard",0),"구역 Jaccard"),
        ("dwell_mape",  result.get("dwell_mape",0), "체류 MAPE"),
    ]:
        metric_row(label, f"{val_num:.3f}", "info")

    score = result.get("overall_score", 0)
    ok = score >= gate["score"]
    if not ok: passed = False
    metric_row("종합 점수", f"{score:.3f}",
               "pass" if ok else "fail", f"≥{gate['score']}")

    stage_footer(passed, time.time()-t0, n, total)
    return passed, result


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CCTV 분석 파이프라인")
    p.add_argument("--day",            default="10",  help="분석 대상 일차")
    p.add_argument("--display",        action="store_true", default=False,
                   help="영상 실시간 표시 (느려짐)")
    p.add_argument("--skip-mosaic",    action="store_true")
    p.add_argument("--skip-validate",  action="store_true")
    p.add_argument("--force",          action="store_true",
                   help="FAIL이어도 다음 단계 강제 진행")
    args = p.parse_args()

    total_stages = 6 - (1 if args.skip_mosaic else 0) - (1 if args.skip_validate else 0)
    sn = 0   # stage number

    print(f"\n{'═'*W}")
    print(f"  CCTV 분석 파이프라인  |  day={args.day}  |  {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'═'*W}")

    STAGE_LABELS = {
        "data":     "데이터 확인",
        "mosaic":   "얼굴 모자이크",
        "zones":    "구역 설정 확인",
        "analyze":  "영상 분석",
        "export":   "CSV 출력",
        "validate": "결과 검증",
    }

    run_log = {"day": args.day, "started": datetime.now().isoformat(), "stages": []}
    pipeline_t0 = time.time()

    def run_stage(fn, label, *a):
        nonlocal sn
        sn += 1
        t0_stage = time.time()
        ok, meta = fn(*a)
        elapsed_stage = round(time.time() - t0_stage, 1)
        run_log["stages"].append({
            "stage": label, "label": STAGE_LABELS.get(label, label),
            "passed": ok, "elapsed_sec": elapsed_stage, "meta": meta,
        })
        if not ok and not args.force:
            print(f"\n  ❌ 파이프라인 중단 (--force 로 강제 진행 가능)")
            save_log(run_log)
            sys.exit(1)
        return ok, meta

    # 단계 실행
    run_stage(lambda: stage_data(args.day, sn+1, total_stages),  "data")
    if not args.skip_mosaic:
        run_stage(lambda: stage_mosaic(args.day, sn+1, total_stages), "mosaic")
    run_stage(lambda: stage_zones(args.day, sn+1, total_stages),  "zones")
    run_stage(lambda: stage_analyze(args.day, args.display, sn+1, total_stages), "analyze")
    run_stage(lambda: stage_export(sn+1, total_stages),           "export")
    if not args.skip_validate:
        run_stage(lambda: stage_validate(sn+1, total_stages),     "validate")

    # ── 단계별 소요 시간 요약표 ──────────────────────────────────────────────
    elapsed_total = time.time() - pipeline_t0
    all_passed    = all(s["passed"] for s in run_log["stages"])

    print(f"\n{'═'*W}")
    print(f"  단계별 소요 시간")
    print(f"{'─'*W}")
    print(f"  {'단계':<26}{'소요시간':>10}  {'결과':<8}  비율")
    print(f"  {'─'*W}")
    for s in run_log["stages"]:
        mark = "✅ PASS" if s["passed"] else "❌ FAIL"
        pct  = s["elapsed_sec"] / elapsed_total * 100 if elapsed_total > 0 else 0
        bar  = "█" * int(pct / 5)   # 5%당 1칸
        print(f"  {s['label']:<26}{s['elapsed_sec']:>8.1f}초  {mark:<8}  {bar} {pct:.1f}%")
    print(f"{'─'*W}")
    print(f"  {'합계':<26}{elapsed_total:>8.1f}초  ({elapsed_total/60:.1f}분)")
    print(f"{'═'*W}")
    print(f"  {'✅ 전 단계 PASS' if all_passed else '⚠️  일부 FAIL (--force 로 진행됨)'}")
    print(f"{'═'*W}\n")

    run_log["finished"]     = datetime.now().isoformat()
    run_log["total_sec"]    = round(elapsed_total, 1)
    run_log["all_passed"]   = all_passed
    save_log(run_log)


def save_log(run_log):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOGS_DIR / f"run_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, ensure_ascii=False, indent=2, default=str)
    print(f"  실행 로그 저장: {path.name}")


if __name__ == "__main__":
    main()
