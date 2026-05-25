"""
main.py - 매장 CCTV 분석 통합 파이프라인

파이프라인:
  1. YOLOv8n-pose + ByteTrack → 사람 검출 + 추적 + 관절 좌표
  2. Pose 키 측정 (머리~발목) + 2점 자동 원근 보정 → 성년/미성년
  3. MiVOLO v2 → 성별 (남/여) 다수결 투표
  4. 최종 4클래스: 성년남성, 성년여성, 미성년남성, 미성년여성
  5. 클래스별 체류 히트맵 생성 + 추적 좌표 CSV 저장

사용법:
    python main.py --source data/test1.mp4 --save-video
    python main.py --source data/test2.mp4 --save-video --show
"""

import argparse
import os
import sys
import csv
import cv2
import numpy as np
import torch
from collections import Counter, defaultdict
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from heatmap_generator import HeatmapGenerator

# ══════════════════════════════════════════════
# MiVOLO v2 로드
# ══════════════════════════════════════════════
print("[정보] MiVOLO v2 모델 로딩중...")
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

CONFIG = AutoConfig.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
MIVOLO_MODEL = AutoModelForImageClassification.from_pretrained(
    "iitolstykh/mivolo_v2", trust_remote_code=True, torch_dtype=torch.float32
)
IMAGE_PROCESSOR = AutoImageProcessor.from_pretrained(
    "iitolstykh/mivolo_v2", trust_remote_code=True
)
MIVOLO_MODEL.eval()
print("[정보] MiVOLO v2 로딩 완료!")

# ══════════════════════════════════════════════
# 4클래스
# ══════════════════════════════════════════════
CLASS_LABELS = {
    "adult_male": "성년 남성",
    "adult_female": "성년 여성",
    "minor_male": "미성년 남성",
    "minor_female": "미성년 여성",
}
CLASS_COLORS = {
    "adult_male": (255, 150, 0),
    "adult_female": (147, 20, 255),
    "minor_male": (0, 200, 0),
    "minor_female": (0, 165, 255),
}
UNCLASSIFIED_COLOR = (180, 180, 180)

SKELETON_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16),
]


# ══════════════════════════════════════════════
# 포즈 기반 키 측정
# ══════════════════════════════════════════════
def estimate_height_from_keypoints(kp, conf_thr=0.3):
    """관절 좌표에서 키(pixel) 측정"""
    # 발목 y
    ankle_y = None
    if kp[15][2] > conf_thr and kp[16][2] > conf_thr:
        ankle_y = max(kp[15][1], kp[16][1])
    elif kp[15][2] > conf_thr:
        ankle_y = kp[15][1]
    elif kp[16][2] > conf_thr:
        ankle_y = kp[16][1]

    if ankle_y is None:
        # 무릎으로 발목 추정
        knee_y, hip_y = None, None
        for i in [13, 14]:
            if kp[i][2] > conf_thr:
                knee_y = kp[i][1] if knee_y is None else max(knee_y, kp[i][1])
        for i in [11, 12]:
            if kp[i][2] > conf_thr:
                hip_y = kp[i][1] if hip_y is None else max(hip_y, kp[i][1])
        if knee_y is not None and hip_y is not None:
            ankle_y = knee_y + (knee_y - hip_y) * 1.1

    if ankle_y is None:
        return None

    # 머리 y
    head_y = None
    for i in [0, 1, 2, 3, 4]:
        if kp[i][2] > conf_thr:
            head_y = kp[i][1] if head_y is None else min(head_y, kp[i][1])
            if i == 0:
                break  # 코가 있으면 최우선

    if head_y is not None:
        return max(0, (ankle_y - head_y) * 1.15)

    # 어깨만 보이는 경우
    shoulder_y = None
    for i in [5, 6]:
        if kp[i][2] > conf_thr:
            shoulder_y = kp[i][1] if shoulder_y is None else min(shoulder_y, kp[i][1])
    if shoulder_y is not None:
        return max(0, (ankle_y - shoulder_y) * 1.35)

    return None


def get_kp_quality(kp, conf_thr=0.3):
    n = sum(1 for i in range(17) if kp[i][2] > conf_thr)
    has_head = any(kp[i][2] > conf_thr for i in [0,1,2])
    has_ankle = any(kp[i][2] > conf_thr for i in [15,16])
    has_knee = any(kp[i][2] > conf_thr for i in [13,14])
    if has_head and has_ankle and n >= 10: return "A"
    elif has_head and has_knee: return "B"
    elif has_ankle or has_knee: return "C"
    return "D"


# ══════════════════════════════════════════════
# 2점 자동 보정
# ══════════════════════════════════════════════
class TwoPointCalibrator:
    def __init__(self, H):
        self.H = H
        self.data = []
        self.slope = 0.0
        self.intercept = 0.0
        self.calibrated = False
        self.ref_height = None
        self.auto_threshold = None

    def add_sample(self, y_center, raw_h, quality="A"):
        if quality == "D" or raw_h < 30:
            return
        self.data.append((y_center, raw_h))
        if len(self.data) >= 20 and not self.calibrated:
            self._calibrate()
        elif self.calibrated and len(self.data) % 100 == 0:
            self._calibrate()

    def _calibrate(self):
        ys = np.array([p[0] for p in self.data])
        hs = np.array([p[1] for p in self.data])
        if np.std(ys) < 10:
            return
        self.slope, self.intercept = np.polyfit(ys, hs, 1)
        self.ref_height = self.slope * self.H + self.intercept
        if self.ref_height <= 0:
            self.ref_height = float(np.median(hs))
        self.calibrated = True
        self.auto_threshold = float(np.median(hs)) * 0.6
        print(f"  [보정] 캘리브레이션 완료 ({len(self.data)}샘플), "
              f"기준={self.ref_height:.0f}px, 추천임계값={self.auto_threshold:.0f}px")

    def correct(self, raw_h, y_center):
        if not self.calibrated:
            return raw_h
        exp = self.slope * y_center + self.intercept
        if exp > 0 and self.ref_height > 0:
            return max(0, raw_h * self.ref_height / exp)
        return raw_h

    def get_threshold(self):
        return self.auto_threshold

    def get_info(self):
        if self.calibrated:
            return f"Auto ({len(self.data)} samples, thr={self.auto_threshold:.0f}px)"
        return f"Collecting ({len(self.data)}/20)"


# ══════════════════════════════════════════════
# MiVOLO 성별 추정
# ══════════════════════════════════════════════
def predict_gender(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    pad_x, pad_y = int((x2-x1)*0.05), int((y2-y1)*0.05)
    body = frame[max(0,y1-pad_y):min(h,y2+pad_y), max(0,x1-pad_x):min(w,x2+pad_x)]
    if body.size == 0: return None
    face_h = (y2-y1)//3
    face = frame[max(0,y1):min(h,y1+face_h), max(0,x1):min(w,x2)]
    try:
        bi = IMAGE_PROCESSOR(images=[body])["pixel_values"].to(dtype=MIVOLO_MODEL.dtype)
        if face.size > 0 and face.shape[0] >= 10 and face.shape[1] >= 10:
            fi = IMAGE_PROCESSOR(images=[face])["pixel_values"].to(dtype=MIVOLO_MODEL.dtype)
        else:
            fi = torch.zeros_like(bi)
        with torch.no_grad():
            out = MIVOLO_MODEL(faces_input=fi, body_input=bi)
        gid = out.gender_class_idx[0].item()
        return {"gender": CONFIG.gender_id2label[gid],
                "prob": round(out.gender_probs[0].item()*100, 1)}
    except: return None


# ══════════════════════════════════════════════
# 성별 투표
# ══════════════════════════════════════════════
class GenderVoter:
    def __init__(self, min_votes=3, min_conf=55.0):
        self.votes = defaultdict(list)
        self.finalized = {}
        self.mv, self.mc = min_votes, min_conf

    def add(self, tid, gender, prob):
        if tid in self.finalized: return
        if prob >= self.mc:
            self.votes[tid].append((gender, prob))
        if len(self.votes[tid]) >= self.mv:
            gs = [v[0] for v in self.votes[tid]]
            self.finalized[tid] = Counter(gs).most_common(1)[0][0]
            g_kr = "남성" if self.finalized[tid] == "male" else "여성"
            print(f"  ✓ ID {tid} 성별: {g_kr}")

    def get(self, tid): return self.finalized.get(tid)
    def tentative(self, tid):
        if tid in self.finalized: return self.finalized[tid]
        v = self.votes.get(tid, [])
        return Counter([x[0] for x in v]).most_common(1)[0][0] if v else None
    def needs(self, tid): return tid not in self.finalized


# ══════════════════════════════════════════════
# 한글 렌더링
# ══════════════════════════════════════════════
_FC = {}
def get_font(sz=18):
    if sz in _FC: return _FC[sz]
    for fp in ["C:/Windows/Fonts/malgun.ttf","C:/Windows/Fonts/gulim.ttc",
               "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
               "/System/Library/Fonts/AppleGothic.ttf"]:
        if os.path.exists(fp):
            try: _FC[sz]=ImageFont.truetype(fp,sz); return _FC[sz]
            except: continue
    _FC[sz]=ImageFont.load_default(); return _FC[sz]

def put_kr(f,t,p,sz=18,c=(255,255,255),bg=None):
    pil=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    d=ImageDraw.Draw(pil); fn=get_font(sz); rc=(c[2],c[1],c[0])
    if bg:
        rb=(bg[2],bg[1],bg[0]); bb=d.textbbox(p,t,font=fn)
        d.rectangle([bb[0]-2,bb[1]-2,bb[2]+2,bb[3]+2],fill=rb)
    d.text(p,t,font=fn,fill=rc)
    np.copyto(f,cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR))

def draw_skeleton(f, kp, color, thr=0.3):
    for i in range(17):
        if kp[i][2]>thr: cv2.circle(f,(int(kp[i][0]),int(kp[i][1])),3,color,-1)
    for i,j in SKELETON_CONNECTIONS:
        if kp[i][2]>thr and kp[j][2]>thr:
            cv2.line(f,(int(kp[i][0]),int(kp[i][1])),(int(kp[j][0]),int(kp[j][1])),color,1)


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="매장 CCTV 분석 파이프라인")
    parser.add_argument("--source", type=str, default="data/test1.mp4")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--skip-frames", type=int, default=5)
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--min-conf", type=float, default=55.0)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--height-threshold", type=int, default=90,
                        help="성년/미성년 임계값 (자동보정 전 폴백)")
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"[오류] 파일 없음: {args.source}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    source_name = os.path.splitext(os.path.basename(args.source))[0]

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("=" * 60)
    print("매장 CCTV 분석 파이프라인")
    print("매장 CCTV 영상에서의 실시간 고객 속성 분류 및")
    print("체류 히트맵 기반 상품 배치 최적화")
    print("=" * 60)
    print(f"입력: {args.source} ({W}x{H}, {fps:.0f}fps, {total}프레임)")
    print("=" * 60)

    # ── 모듈 초기화 ──
    print("\n[로드] YOLOv8n-pose...")
    yolo = YOLO("yolov8n-pose.pt")

    calibrator = TwoPointCalibrator(H)
    voter = GenderVoter(min_votes=args.min_votes, min_conf=args.min_conf)
    attempt_count = defaultdict(int)

    heatmap_gen = HeatmapGenerator(
        width=W, height=H, resolution_scale=1.0, class_labels=CLASS_LABELS
    )

    # 비디오 라이터
    writer = None
    if args.save_video:
        out_path = os.path.join(args.output_dir, f"tracked_{source_name}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print("[처리] 프레임 처리중...\n")
    cap = cv2.VideoCapture(args.source)
    frame_idx = 0

    id_heights = defaultdict(list)
    id_final_class = {}
    current_tracks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── 1. YOLOv8n-pose + ByteTrack ──
        results = yolo.track(
            frame, conf=args.conf,
            tracker="bytetrack.yaml", persist=True, verbose=False
        )

        current_tracks = []
        boxes = results[0].boxes
        kps_data = results[0].keypoints

        if boxes is not None and boxes.id is not None and kps_data is not None:
            xyxys = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            all_kps = kps_data.data.cpu().numpy()

            for idx, (box, tid) in enumerate(zip(xyxys, ids)):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                kps = all_kps[idx]
                bbox_h = y2 - y1
                y_center = (y1 + y2) / 2

                # ── 2. 키 측정 ──
                raw_h = estimate_height_from_keypoints(kps)
                quality = get_kp_quality(kps)
                if raw_h is None:
                    raw_h = float(bbox_h)
                    quality = "D"

                calibrator.add_sample(y_center, raw_h, quality)
                corrected_h = calibrator.correct(raw_h, y_center)

                current_tracks.append((tid, x1, y1, x2, y2, kps, raw_h, corrected_h, quality))
                id_heights[tid].append((raw_h, corrected_h, y_center, quality))

                # ── 3. 히트맵 좌표 누적 (확정된 ID만) ──
                if tid in id_final_class:
                    cx = (x1 + x2) / 2
                    heatmap_gen.add_point(id_final_class[tid], cx, float(y2))

                # ── 4. 성별 분류 (MiVOLO) ──
                if frame_idx % args.skip_frames != 0: continue
                if not voter.needs(tid): continue
                if attempt_count[tid] >= args.max_attempts: continue
                if bbox_h < 50: continue

                attempt_count[tid] += 1
                result = predict_gender(frame, x1, y1, x2, y2)
                if result:
                    voter.add(tid, result["gender"], result["prob"])

        # ── 5. 최종 4클래스 결정 ──
        eff_thr = calibrator.get_threshold() or args.height_threshold

        for track in current_tracks:
            tid = track[0]
            if tid in id_final_class: continue

            gender = voter.get(tid)
            if gender is None: continue

            heights = id_heights.get(tid, [])
            if len(heights) < 3: continue

            corr_list = [h[1] for h in heights if h[3] != "D" and h[1] > 10]
            if not corr_list:
                corr_list = [h[1] for h in heights if h[1] > 10]
            if not corr_list: continue

            med = np.median(corr_list)
            adult = med >= eff_thr

            if adult and gender == "male": id_final_class[tid] = "adult_male"
            elif adult and gender == "female": id_final_class[tid] = "adult_female"
            elif not adult and gender == "male": id_final_class[tid] = "minor_male"
            else: id_final_class[tid] = "minor_female"

            raw_med = np.median([h[0] for h in heights if h[0] > 10])
            print(f"  ★ ID {tid}: {CLASS_LABELS[id_final_class[tid]]} "
                  f"(원본={raw_med:.0f}px, 보정={med:.0f}px, 임계={eff_thr:.0f}px)")

        # ── 시각화 ──
        if args.save_video or args.show:
            frame_vis = frame.copy()
            debug_frame = frame.copy() if args.show else None

            for track in current_tracks:
                tid, x1, y1, x2, y2, kps, raw_h, corr_h, quality = track
                cls = id_final_class.get(tid)

                # 메인 영상
                if cls:
                    color = CLASS_COLORS[cls]
                    label = f"ID{tid} {CLASS_LABELS[cls]}"
                else:
                    tg = voter.tentative(tid)
                    color = UNCLASSIFIED_COLOR
                    gs = "남?" if tg=="male" else ("여?" if tg=="female" else "?")
                    label = f"ID{tid} {gs}"

                cv2.rectangle(frame_vis, (x1,y1), (x2,y2), color, 2)
                put_kr(frame_vis, label, (x1, max(0,y1-22)), sz=15, c=(255,255,255), bg=color)

                # 디버그 영상
                if debug_frame is not None:
                    is_ad = corr_h >= eff_thr
                    dc = (0,255,0) if is_ad else (0,255,255)
                    draw_skeleton(debug_frame, kps, dc)
                    cv2.rectangle(debug_frame, (x1,y1),(x2,y2), dc, 1)
                    tag = "Adult" if is_ad else "Minor"
                    cv2.putText(debug_frame,
                                f"ID{tid}[{quality}] {raw_h:.0f}->{corr_h:.0f}px [{tag}]",
                                (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dc, 1)

            # 범례
            ov = frame_vis.copy()
            cv2.rectangle(ov, (5,5), (200,130), (0,0,0), -1)
            cv2.addWeighted(ov, 0.6, frame_vis, 0.4, 0, frame_vis)
            for i,(ck,nm) in enumerate(CLASS_LABELS.items()):
                yp = 15+i*25
                cv2.rectangle(frame_vis, (10,yp),(25,yp+15), CLASS_COLORS[ck], -1)
                put_kr(frame_vis, nm, (32,yp-2), sz=14, c=(255,255,255))

            if writer: writer.write(frame_vis)
            if args.show:
                ci = calibrator.get_info()
                cv2.putText(debug_frame, f"Thr: {eff_thr:.0f}px | {ci}",
                            (10,H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
                cv2.imshow("[Debug] Pose + Height", debug_frame)
                cv2.imshow("[Final] Classification", frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  프레임 {frame_idx}/{total} | "
                  f"추적 {len(current_tracks)}명 | 확정 {len(id_final_class)}명")

    cap.release()
    if writer:
        writer.release()
        print(f"\n  → 추적 영상: {out_path}")
    if args.show:
        cv2.destroyAllWindows()

    # ── 히트맵 생성 ──
    print("\n[히트맵] 생성중...")

    for cls in CLASS_LABELS:
        p = os.path.join(args.output_dir, f"{source_name}_heatmap_{cls}.png")
        heatmap_gen.save_individual_heatmap(cls, p)
        print(f"  → {CLASS_LABELS[cls]}: {p}")

    p = os.path.join(args.output_dir, f"{source_name}_heatmap_combined.png")
    heatmap_gen.save_combined_heatmap(p)
    print(f"  → 통합: {p}")

    p = os.path.join(args.output_dir, f"{source_name}_heatmap_comparison.png")
    heatmap_gen.save_comparison_heatmap(p)
    print(f"  → 비교: {p}")

    # ── CSV 저장 ──
    csv_path = os.path.join(args.output_dir, f"{source_name}_tracking.csv")
    print(f"\n[CSV] 저장중: {csv_path}")
    # 여기서는 id_heights 데이터를 활용
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["영상파일", "추적ID", "분류코드", "분류명",
                     "보정키_중간값", "원본키_중간값", "관절품질"])
        for tid in sorted(id_heights.keys()):
            cls = id_final_class.get(tid, "미분류")
            cls_name = CLASS_LABELS.get(cls, "미분류")
            hs = id_heights[tid]
            corrs = [h[1] for h in hs if h[1] > 10]
            raws = [h[0] for h in hs if h[0] > 10]
            best_q = min([h[3] for h in hs], key=lambda x: "ABCD".index(x))
            w.writerow([
                source_name, tid, cls, cls_name,
                f"{np.median(corrs):.1f}" if corrs else "N/A",
                f"{np.median(raws):.1f}" if raws else "N/A",
                best_q
            ])

    # ── 최종 결과 ──
    total_ids = len(set(id_heights.keys()))
    classified = len(id_final_class)
    counts = Counter(id_final_class.values())

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"  총 고유 ID: {total_ids}명")
    print(f"  분류 확정:  {classified}명 ({classified/max(total_ids,1)*100:.0f}%)")
    if calibrator.calibrated:
        print(f"  원근 보정:  {calibrator.get_info()}")
    print()
    for ck, nm in CLASS_LABELS.items():
        c = counts.get(ck, 0)
        print(f"  {nm}: {c}명 ({c/max(classified,1)*100:.1f}%)")

    all_raw = [np.median([h[0] for h in hs if h[0]>10])
               for hs in id_heights.values()
               if any(h[0]>10 for h in hs)]
    all_corr = [np.median([h[1] for h in hs if h[1]>10])
                for hs in id_heights.values()
                if any(h[1]>10 for h in hs)]
    if all_raw:
        print(f"\n  원본 키: {min(all_raw):.0f}~{max(all_raw):.0f}px (avg={np.mean(all_raw):.0f})")
        print(f"  보정 키: {min(all_corr):.0f}~{max(all_corr):.0f}px (avg={np.mean(all_corr):.0f})")
    print("=" * 60)


if __name__ == "__main__":
    main()