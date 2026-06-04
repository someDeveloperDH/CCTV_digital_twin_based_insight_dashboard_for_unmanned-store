"""
매대 구역 설정 도구 (카메라별)

각 카메라에서 보이는 구역만 폴리곤 설정.
같은 구역이 여러 카메라에 걸쳐 보일 수 있고, 어떤 카메라에는 안 보일 수 있음.

zones.json 저장 구조:
  "polygons": {"ch3": [[x,y],...], "ch6": [[x,y],...], "ch1": [...]}

실행: python zone_config.py [--cameras ch3,ch6,ch1] [--day 10]
키:   클릭=꼭짓점추가  E=구역완료  S=이카메라에안보임  U=되돌리기  D=이전구역다시  Q=저장후다음카메라
"""
import cv2
import json
import argparse
import numpy as np
from pathlib import Path

BASE_DIR   = Path(__file__).parent
MOSAIC_DIR = BASE_DIR / "mosaic"
MERGED_DIR = BASE_DIR / "merged"
ZONES_FILE = BASE_DIR / "zones.json"

COLORS = [
    (0,255,0),(255,128,0),(0,128,255),(255,0,128),
    (128,255,0),(0,255,200),(200,0,255),(255,255,0),(0,128,128)
]

# ── 전역 상태 ─────────────────────────────────────────────────────────────────
state = {
    "zones":           [],
    "current_idx":     0,
    "current_polygon": [],
    "frame_orig":      None,
    "frame_display":   None,
    "cam":             "",
}


def load_zones():
    with open(ZONES_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_zones(zones):
    with open(ZONES_FILE, "w", encoding="utf-8") as f:
        json.dump(zones, f, ensure_ascii=False, indent=2)


def find_video(cam, day):
    """mosaic 우선, 없으면 merged 사용"""
    for d in [MOSAIC_DIR, MERGED_DIR]:
        p = d / f"{cam}_{day}.mp4"
        if p.exists():
            return p
    return None


def draw_frame():
    img = state["frame_orig"].copy()
    h, w = img.shape[:2]
    zones = state["zones"]
    cam   = state["cam"]
    idx   = state["current_idx"]
    ov    = img.copy()

    # 이미 설정된 구역 표시
    for i, z in enumerate(zones):
        pts = z.get("polygons", {}).get(cam, [])
        col = COLORS[i % len(COLORS)]
        if isinstance(pts, list) and len(pts) >= 3:
            poly = np.array(pts, dtype=np.int32)
            cv2.fillPoly(ov, [poly], col)
            cv2.polylines(img, [poly], True, col, 2)
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
            cv2.putText(img, z["name"], (cx-50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)

    # 현재 그리는 폴리곤
    cur = state["current_polygon"]
    if cur:
        for pt in cur:
            cv2.circle(img, pt, 5, (0, 0, 255), -1)
        if len(cur) >= 2:
            cv2.polylines(img, [np.array(cur)], False, (0, 0, 255), 2)

    # 상단 가이드 바
    if idx < len(zones):
        z   = zones[idx]
        col = COLORS[idx % len(COLORS)]
        already = z.get("polygons", {}).get(cam, [])
        if isinstance(already, list) and len(already) >= 3:
            status = "✅완료"
        elif already == "skip":
            status = "⏭건너뜀"
        else:
            status = "미설정"
        guide = (f"[{cam}] [{idx+1}/{len(zones)}] {z['name']}  {status}  "
                 f"| 상품: {','.join(z['products'][:2])}... | {z['price']}원")
    else:
        guide = f"[{cam}] 모든 구역 완료! Q를 눌러 다음 카메라로"
        col   = (0, 200, 100)

    cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(img, guide, (6, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

    # 하단 조작 안내
    cv2.rectangle(img, (0, h-30), (w, h), (0, 0, 0), -1)
    cv2.putText(img,
                "클릭:점추가  E:구역완료  S:안보임(skip)  U:되돌리기  D:이전구역다시  Q:저장후다음카메라",
                (6, h-9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    state["frame_display"] = img


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if state["current_idx"] < len(state["zones"]):
            state["current_polygon"].append((x, y))
            draw_frame()


def select_visible_zones(cam, zones):
    """
    터미널에서 이 카메라에 보이는 구역을 번호로 선택.
    선택된 인덱스 목록 반환.
    선택되지 않은 구역은 자동으로 'skip' 처리됨.
    """
    # 카메라별 안내 힌트
    hints = {
        "ch1": "ch1은 입구·계산대를 바라봅니다 → 출입문, 계산대 + 입구에서 보이는 구역",
        "ch3": "ch3은 매장 앞쪽을 바라봅니다 → 아이스크림·음료·과자 일부",
        "ch6": "ch6은 매장 뒤쪽을 바라봅니다 → 아이스크림·음료·과자 일부",
    }
    print(f"\n{'─'*60}")
    print(f"  [{cam}] 이 카메라에서 보이는 구역 선택")
    print(f"  힌트: {hints.get(cam, '')}")
    print(f"{'─'*60}")
    for i, z in enumerate(zones, 1):
        poly = z.get("polygons", {}).get(cam, [])
        if isinstance(poly, list) and len(poly) >= 3:
            status = "✅ 완료"
        elif poly == "skip":
            status = "⏭ skip"
        else:
            status = "⬜ 미설정"
        prod = f"  ({','.join(z['products'][:2])}...)" if z['products'] else ""
        print(f"  {i:2d}. {z['name']:<12} {status}{prod}")
    print(f"{'─'*60}")
    print(f"  보이는 구역 번호 입력 (예: 1,2,4,10)")
    print(f"  all=전체선택  enter=이미설정된것만  skip=전체건너뜀")

    while True:
        try:
            raw = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []

        if raw == "all":
            return list(range(len(zones)))
        if raw == "skip":
            return []
        if raw == "":
            # 이미 완료된 것만
            return [i for i, z in enumerate(zones)
                    if isinstance(z.get("polygons",{}).get(cam,[]), list)
                    and len(z.get("polygons",{}).get(cam,[])) >= 3]
        try:
            idxs = [int(x.strip()) - 1 for x in raw.split(",") if x.strip()]
            if all(0 <= n < len(zones) for n in idxs):
                return idxs
            print(f"  1~{len(zones)} 범위의 번호를 입력하세요")
        except ValueError:
            print(f"  숫자만 입력하세요 (예: 1,3,5)")


def configure_camera(cam, day, zones):
    """한 카메라에 대해 구역 설정 후 zones 반환."""
    video_path = find_video(cam, day)
    if video_path is None:
        print(f"  [{cam}] 영상 없음 ({cam}_{day}.mp4) → 건너뜀")
        return zones

    print(f"\n  [{cam}] 영상: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  [{cam}] 첫 프레임 읽기 실패")
        return zones

    # ── 구역 사전 선택 ────────────────────────────────────────────────────────
    selected_idxs = select_visible_zones(cam, zones)

    # 선택 안 된 구역은 이 카메라에서 skip 처리
    for i, z in enumerate(zones):
        if i not in selected_idxs:
            # 이미 완료된 것은 건드리지 않음
            existing = z.get("polygons", {}).get(cam, [])
            if not (isinstance(existing, list) and len(existing) >= 3):
                z.setdefault("polygons", {})[cam] = "skip"

    if not selected_idxs:
        print(f"  [{cam}] 선택된 구역 없음 → 건너뜀")
        return zones

    print(f"\n  [{cam}] 그릴 구역: {[zones[i]['name'] for i in selected_idxs]}")

    # selected_idxs 순서대로 그리기 위해 state에 필터된 구역 목록 사용
    zones_to_draw = [zones[i] for i in selected_idxs
                     if not (isinstance(zones[i].get("polygons",{}).get(cam,[]), list)
                             and len(zones[i].get("polygons",{}).get(cam,[])) >= 3)]

    if not zones_to_draw:
        print(f"  [{cam}] 선택한 구역 모두 이미 완료됨")
        return zones

    state["zones"]           = zones_to_draw
    state["current_idx"]     = 0
    state["current_polygon"] = []
    state["frame_orig"]      = frame.copy()
    state["cam"]             = cam
    draw_frame()

    win = f"ZoneConfig-{cam}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, state["frame_display"])

    # Qt가 창을 실제로 생성할 때까지 대기 (최대 3초 재시도)
    registered = False
    for _ in range(30):
        cv2.waitKey(100)
        try:
            cv2.setMouseCallback(win, mouse_cb)
            registered = True
            break
        except cv2.error:
            pass
    if not registered:
        print(f"  [{cam}] 창 생성 실패 — DISPLAY 환경변수 확인 필요")
        cv2.destroyAllWindows()
        return zones

    cv2.resizeWindow(win, 1280, 720)

    print(f"  구역 수: {len(zones)}개 | 시작: {start_idx+1}번째")
    print(f"  클릭=점추가  E=완료  S=안보임  U=되돌리기  D=이전  Q=저장후다음\n")

    while True:
        cv2.imshow(win, state["frame_display"])
        key = cv2.waitKey(20) & 0xFF
        idx = state["current_idx"]

        cur_zones = state["zones"]
        if key in (ord('e'), ord('E')):
            cur = state["current_polygon"]
            if len(cur) < 3:
                print("  꼭짓점 3개 이상 필요"); continue
            if idx >= len(cur_zones):
                print("  모든 구역 완료"); continue
            cur_zones[idx].setdefault("polygons", {})[cam] = list(cur)
            print(f"  ✅ [{cur_zones[idx]['name']}] {cam} 저장 ({len(cur)}꼭짓점)")
            state["current_polygon"].clear()
            state["current_idx"] += 1
            draw_frame()

        elif key in (ord('s'), ord('S')):
            if idx < len(cur_zones):
                cur_zones[idx].setdefault("polygons", {})[cam] = "skip"
                print(f"  ⏭ [{cur_zones[idx]['name']}] {cam} — 안 보임(skip)")
                state["current_polygon"].clear()
                state["current_idx"] += 1
                draw_frame()

        elif key in (ord('u'), ord('U')):
            if state["current_polygon"]:
                state["current_polygon"].pop()
                draw_frame()

        elif key in (ord('d'), ord('D')):
            if state["current_idx"] > 0:
                state["current_idx"] -= 1
                cur_zones[state["current_idx"]].setdefault("polygons", {})[cam] = []
                state["current_polygon"].clear()
                print(f"  ↩ [{cur_zones[state['current_idx']]['name']}] 다시 그리기")
                draw_frame()

        elif key in (ord('q'), ord('Q')):
            break

    cv2.destroyWindow(win)

    # 완료 통계는 전체 zones 기준
    done = skip = empty = 0
    for z in zones:
        poly = z.get("polygons", {}).get(cam, [])
        if isinstance(poly, list) and len(poly) >= 3: done += 1
        elif poly == "skip": skip += 1
        else: empty += 1
    print(f"  [{cam}] 완료={done}  건너뜀={skip}  미설정={empty}")
    return zones  # 원본 zones 반환 (zones_to_draw는 같은 객체 참조)


def print_summary(zones, cameras):
    print(f"\n{'─'*60}")
    print(f"  최종 설정 현황")
    print(f"{'─'*60}")
    print(f"  {'구역명':<14}" + "".join(f"  {c:<8}" for c in cameras))
    for z in zones:
        row = f"  {z['name']:<14}"
        for cam in cameras:
            poly = z.get("polygons", {}).get(cam, [])
            if isinstance(poly, list) and len(poly) >= 3:
                row += f"  {'✅':<8}"
            elif poly == "skip":
                row += f"  {'⏭':<8}"
            else:
                row += f"  {'⬜':<8}"
        print(row)
    print(f"{'─'*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", default="ch3,ch6,ch1",
                        help="설정할 카메라 순서 (쉼표 구분, 기본: ch3,ch6,ch1)")
    parser.add_argument("--day", default="10")
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",")]
    zones   = load_zones()

    print(f"구역 수: {len(zones)}개")
    print(f"설정 카메라: {cameras}")
    print(f"영상 day: {args.day}")
    print(f"영상 소스: mosaic 우선, 없으면 merged 사용\n")

    for cam in cameras:
        zones = configure_camera(cam, args.day, zones)
        save_zones(zones)
        print(f"  → zones.json 저장 완료")

    print_summary(zones, cameras)
    print(f"\n모든 카메라 설정 완료 → {ZONES_FILE}")


if __name__ == "__main__":
    main()
