"""
test_camera_geometry.py - 카메라 기하학 기반 키 측정 + 성년/미성년 분류

원리:
  카메라 설치 높이(H), 틸트 각도(tilt), 수직 화각(FOV_V)을 알면
  y픽셀 → 바닥까지 실제 거리(mm) → 해당 위치에서 1px = 몇 mm 를 계산 가능
  → 사람 머리~발 pixel 차이 × mm/px = 실제 키(mm)

  카메라
    ↑H(높이)
    |  ╲ ← tilt 각도
    |   ╲
  ──┴────╲──── 바닥
         d(거리)

  d = H × tan(각도)
  화면 상단(먼곳): 각도 = tilt - FOV_V/2
  화면 중앙:       각도 = tilt
  화면 하단(가까이): 각도 = tilt + FOV_V/2

사용법:
    python test_camera_geometry.py --source data/test1.mp4 --save-video --show

    # 카메라 파라미터 변경
    python test_camera_geometry.py --source data/test1.mp4 --save-video \\
        --cam-height 3000 --cam-tilt 40 --cam-fov-v 60

    # 성년 기준 키 변경 (기본 150cm)
    python test_camera_geometry.py --source data/test1.mp4 --save-video \\
        --adult-height-cm 140
"""

import argparse
import os
import sys
import csv
import math
import cv2
import numpy as np
import torch
from collections import Counter, defaultdict
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from heatmap_generator import HeatmapGenerator

# ══════════════════════════════════════════════
# MiVOLO v2 (성별만)
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
# 카메라 기하학 모델
# ══════════════════════════════════════════════
class CameraGeometry:
    """
    카메라 설치 파라미터로 y픽셀 → 실제 거리 매핑 테이블 생성

    Parameters:
        cam_height_mm: 카메라 설치 높이 (mm)
        cam_tilt_deg:  카메라 틸트 각도 (수평=0, 아래=양수)
        fov_v_deg:     수직 화각 (degree)
        img_height:    영상 세로 해상도 (px)
    """

    def __init__(self, cam_height_mm: float, cam_tilt_deg: float,
                 fov_v_deg: float, img_height: int):
        self.H = cam_height_mm
        self.tilt = cam_tilt_deg
        self.fov_v = fov_v_deg
        self.img_h = img_height

        # 화면 상단/하단의 시선 각도
        self.angle_top = self.tilt - self.fov_v / 2     # 상단 (먼 곳)
        self.angle_bottom = self.tilt + self.fov_v / 2   # 하단 (가까운 곳)

        # y픽셀 → mm/px 매핑 테이블 생성
        self.mm_per_px_table = self._build_table()

        # 디버그 정보 출력
        self._print_info()

    def _angle_for_y(self, y_px: int) -> float:
        """y픽셀 → 해당 시선의 각도 (degree)"""
        # y=0 → 상단(먼곳) = angle_top
        # y=img_h → 하단(가까이) = angle_bottom
        ratio = y_px / self.img_h
        return self.angle_top + (self.angle_bottom - self.angle_top) * ratio

    def _ground_distance_for_y(self, y_px: int) -> float:
        """y픽셀 → 카메라 직하점에서 바닥 수평 거리 (mm)"""
        angle = self._angle_for_y(y_px)
        if angle <= 0 or angle >= 90:
            return float('inf')
        return self.H * math.tan(math.radians(angle))

    def _build_table(self) -> np.ndarray:
        """y픽셀별 mm/px 매핑 테이블 생성"""
        table = np.zeros(self.img_h, dtype=np.float64)

        for y in range(self.img_h):
            # y와 y+1 사이의 실제 거리 차이 → 1px당 mm
            d1 = self._ground_distance_for_y(y)
            d2 = self._ground_distance_for_y(min(y + 1, self.img_h - 1))

            if d1 == float('inf') or d2 == float('inf'):
                table[y] = 0
                continue

            # 바닥 위 수평 방향 mm/px (바닥면 기준)
            ground_mm_per_px = abs(d2 - d1)

            # 수직 방향 mm/px 보정
            # 카메라에서 해당 지점까지의 빗변 거리
            angle = self._angle_for_y(y)
            if angle > 0:
                slant_dist = self.H / math.cos(math.radians(angle))
                # 수직 방향 1px이 실제로 몇 mm인지
                # = 빗변 거리에서 화각 1px분의 각도에 해당하는 호의 길이
                angle_per_px = self.fov_v / self.img_h  # 1px당 각도
                vertical_mm_per_px = slant_dist * math.tan(math.radians(angle_per_px))
                table[y] = vertical_mm_per_px
            else:
                table[y] = 0

        return table

    def estimate_real_height_mm(self, head_y: float, foot_y: float) -> float:
        """
        머리 y좌표와 발 y좌표로 실제 키(mm) 추정

        Args:
            head_y: 머리 y픽셀
            foot_y: 발 y픽셀 (head_y < foot_y)

        Returns:
            추정 키 (mm)
        """
        if head_y >= foot_y:
            return 0

        head_y = max(0, min(int(head_y), self.img_h - 1))
        foot_y = max(0, min(int(foot_y), self.img_h - 1))

        # head_y ~ foot_y 사이 각 픽셀의 mm를 합산
        total_mm = 0
        for y in range(head_y, foot_y):
            total_mm += self.mm_per_px_table[y]

        return total_mm

    def estimate_from_keypoints(self, kps, conf_thr=0.3, bbox_y1=None, bbox_y2=None):
        """
        관절 좌표에서 실제 키(mm) 추정

        Returns:
            (height_mm, quality, head_y, foot_y) 또는 (None, "D", None, None)
        """
        # 발 y
        foot_y = None
        for i in [15, 16]:
            if kps[i][2] > conf_thr:
                foot_y = kps[i][1] if foot_y is None else max(foot_y, kps[i][1])

        if foot_y is None:
            # 무릎에서 추정
            knee_y, hip_y = None, None
            for i in [13, 14]:
                if kps[i][2] > conf_thr:
                    knee_y = kps[i][1] if knee_y is None else max(knee_y, kps[i][1])
            for i in [11, 12]:
                if kps[i][2] > conf_thr:
                    hip_y = kps[i][1] if hip_y is None else max(hip_y, kps[i][1])
            if knee_y and hip_y:
                foot_y = knee_y + (knee_y - hip_y) * 1.1
            elif bbox_y2 is not None:
                foot_y = bbox_y2

        if foot_y is None:
            return None, "D", None, None

        # 머리 y
        head_y = None
        for i in [0, 1, 2, 3, 4]:
            if kps[i][2] > conf_thr:
                head_y = kps[i][1] if head_y is None else min(head_y, kps[i][1])
                if i == 0: break

        quality = "A"
        if head_y is not None:
            # 코→정수리 약 10% 보정
            pixel_height = foot_y - head_y
            head_y = head_y - pixel_height * 0.10
        else:
            # 어깨에서 추정
            shoulder_y = None
            for i in [5, 6]:
                if kps[i][2] > conf_thr:
                    shoulder_y = kps[i][1] if shoulder_y is None else min(shoulder_y, kps[i][1])
            if shoulder_y is not None:
                # 어깨 위 = 전체 키의 약 26%
                body_height = foot_y - shoulder_y
                head_y = shoulder_y - body_height * 0.35
                quality = "B"
            elif bbox_y1 is not None:
                head_y = bbox_y1
                quality = "C"
            else:
                return None, "D", None, None

        height_mm = self.estimate_real_height_mm(head_y, foot_y)
        return height_mm, quality, head_y, foot_y

    def get_distance_at_y(self, y_px: int) -> float:
        """y픽셀에서 카메라까지 바닥 수평 거리 (m)"""
        return self._ground_distance_for_y(y_px) / 1000

    def _print_info(self):
        """카메라 정보 출력"""
        d_top = self._ground_distance_for_y(0)
        d_mid = self._ground_distance_for_y(self.img_h // 2)
        d_bot = self._ground_distance_for_y(self.img_h - 1)

        mm_top = self.mm_per_px_table[0]
        mm_mid = self.mm_per_px_table[self.img_h // 2]
        mm_bot = self.mm_per_px_table[self.img_h - 1]

        print(f"\n  [카메라 기하학]")
        print(f"  설치: 높이={self.H/1000:.1f}m, 틸트={self.tilt}°, 화각={self.fov_v}°")
        print(f"  시선: 상단={self.angle_top:.1f}° ~ 하단={self.angle_bottom:.1f}°")
        print(f"  바닥거리: 상단={d_top/1000:.1f}m, 중앙={d_mid/1000:.1f}m, 하단={d_bot/1000:.1f}m")
        print(f"  mm/px:    상단={mm_top:.1f}, 중앙={mm_mid:.1f}, 하단={mm_bot:.1f}")
        print(f"  → 상단에서 100px 사람 ≈ {mm_top*100/10:.0f}cm")
        print(f"  → 하단에서 100px 사람 ≈ {mm_bot*100/10:.0f}cm")


# ══════════════════════════════════════════════
# MiVOLO 성별 / 투표
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
        if face.size > 0 and face.shape[0]>=10 and face.shape[1]>=10:
            fi = IMAGE_PROCESSOR(images=[face])["pixel_values"].to(dtype=MIVOLO_MODEL.dtype)
        else:
            fi = torch.zeros_like(bi)
        with torch.no_grad():
            out = MIVOLO_MODEL(faces_input=fi, body_input=bi)
        gid = out.gender_class_idx[0].item()
        return {"gender": CONFIG.gender_id2label[gid],
                "prob": round(out.gender_probs[0].item()*100, 1)}
    except: return None

class GenderVoter:
    def __init__(self, min_votes=3, min_conf=55.0):
        self.votes = defaultdict(list)
        self.finalized = {}
        self.mv, self.mc = min_votes, min_conf

    def add(self, tid, gender, prob):
        if tid in self.finalized: return
        if prob >= self.mc: self.votes[tid].append((gender, prob))
        if len(self.votes[tid]) >= self.mv:
            gs = [v[0] for v in self.votes[tid]]
            self.finalized[tid] = Counter(gs).most_common(1)[0][0]
            g_kr = "남성" if self.finalized[tid]=="male" else "여성"
            print(f"  ✓ ID {tid} 성별: {g_kr}")

    def get(self, tid): return self.finalized.get(tid)
    def tentative(self, tid):
        if tid in self.finalized: return self.finalized[tid]
        v = self.votes.get(tid,[])
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
    parser = argparse.ArgumentParser(description="카메라 기하학 기반 키 측정 + 분류")
    parser.add_argument("--source", type=str, default="data/test1.mp4")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--show", action="store_true")

    # 카메라 파라미터
    parser.add_argument("--cam-height", type=float, default=2000,
                        help="카메라 설치 높이 (mm)")
    parser.add_argument("--cam-tilt", type=float, default=45,
                        help="카메라 틸트 각도 (도, 수평=0, 아래=양수)")
    parser.add_argument("--cam-fov-v", type=float, default=70,
                        help="카메라 수직 화각 (도)")

    # 분류 설정
    parser.add_argument("--adult-height-cm", type=float, default=150,
                        help="성년 기준 키 (cm, 이 이상이면 성년)")
    parser.add_argument("--skip-frames", type=int, default=5)
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--min-conf", type=float, default=55.0)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"[오류] 파일 없음: {args.source}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    source_name = os.path.splitext(os.path.basename(args.source))[0]
    adult_threshold_mm = args.adult_height_cm * 10  # cm → mm

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print("=" * 60)
    print("카메라 기하학 기반 키 측정 파이프라인")
    print("=" * 60)
    print(f"입력: {args.source} ({W}x{H}, {fps:.0f}fps, {total}프레임)")
    print(f"성년 기준: {args.adult_height_cm}cm 이상")

    # 카메라 기하학 모델 초기화
    cam_geo = CameraGeometry(
        cam_height_mm=args.cam_height,
        cam_tilt_deg=args.cam_tilt,
        fov_v_deg=args.cam_fov_v,
        img_height=H
    )

    print("=" * 60)

    # 모듈 초기화
    print("\n[로드] YOLOv8n-pose...")
    yolo = YOLO("yolov8n-pose.pt")
    voter = GenderVoter(min_votes=args.min_votes, min_conf=args.min_conf)
    attempt_count = defaultdict(int)

    heatmap_gen = HeatmapGenerator(
        width=W, height=H, resolution_scale=1.0, class_labels=CLASS_LABELS
    )

    writer = None
    if args.save_video:
        out_path = os.path.join(args.output_dir, f"camgeo_{source_name}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print("[처리] 프레임 처리중...\n")
    cap = cv2.VideoCapture(args.source)
    frame_idx = 0

    id_heights_mm = defaultdict(list)  # {tid: [(height_mm, quality), ...]}
    id_final_class = {}
    current_tracks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLOv8n-pose + ByteTrack ──
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

                # ── 카메라 기하학 기반 키 측정 (mm) ──
                height_mm, quality, head_y, foot_y = cam_geo.estimate_from_keypoints(
                    kps, bbox_y1=y1, bbox_y2=y2
                )

                if height_mm is None:
                    height_mm = 0
                    quality = "D"

                height_cm = height_mm / 10
                current_tracks.append((tid, x1, y1, x2, y2, kps,
                                       height_mm, height_cm, quality, head_y, foot_y))

                if quality != "D" and height_mm > 0:
                    id_heights_mm[tid].append((height_mm, quality))

                # 히트맵 좌표 누적
                if tid in id_final_class:
                    heatmap_gen.add_point(id_final_class[tid], (x1+x2)/2, float(y2))

                # ── 성별 분류 ──
                if frame_idx % args.skip_frames != 0: continue
                if not voter.needs(tid): continue
                if attempt_count[tid] >= args.max_attempts: continue
                if (y2-y1) < 50: continue

                attempt_count[tid] += 1
                result = predict_gender(frame, x1, y1, x2, y2)
                if result:
                    voter.add(tid, result["gender"], result["prob"])

        # ── 최종 분류 ──
        for track in current_tracks:
            tid = track[0]
            if tid in id_final_class: continue

            gender = voter.get(tid)
            if gender is None: continue

            hs = id_heights_mm.get(tid, [])
            valid = [h for h, q in hs if q != "D" and h > 100]
            if len(valid) < 3: continue

            median_mm = np.median(valid)
            adult = median_mm >= adult_threshold_mm

            if adult and gender == "male": id_final_class[tid] = "adult_male"
            elif adult and gender == "female": id_final_class[tid] = "adult_female"
            elif not adult and gender == "male": id_final_class[tid] = "minor_male"
            else: id_final_class[tid] = "minor_female"

            print(f"  ★ ID {tid}: {CLASS_LABELS[id_final_class[tid]]} "
                  f"(키={median_mm/10:.0f}cm, 기준={args.adult_height_cm}cm, "
                  f"성년={'O' if adult else 'X'})")

        # ── 시각화 ──
        if args.save_video or args.show:
            frame_vis = frame.copy()
            debug_frame = frame.copy() if args.show else None

            for track in current_tracks:
                tid, x1, y1, x2, y2, kps, h_mm, h_cm, quality, head_y, foot_y = track
                cls = id_final_class.get(tid)

                # 메인 영상
                if cls:
                    color = CLASS_COLORS[cls]
                    label = f"ID{tid} {CLASS_LABELS[cls]} ({h_cm:.0f}cm)"
                else:
                    tg = voter.tentative(tid)
                    color = UNCLASSIFIED_COLOR
                    gs = "남?" if tg=="male" else ("여?" if tg=="female" else "?")
                    label = f"ID{tid} {gs} ({h_cm:.0f}cm)"

                cv2.rectangle(frame_vis, (x1,y1),(x2,y2), color, 2)
                put_kr(frame_vis, label, (x1, max(0,y1-22)), sz=15, c=(255,255,255), bg=color)

                # 디버그 영상
                if debug_frame is not None:
                    adult = h_mm >= adult_threshold_mm
                    dc = (0,255,0) if adult else (0,255,255)
                    draw_skeleton(debug_frame, kps, dc)
                    cv2.rectangle(debug_frame, (x1,y1),(x2,y2), dc, 1)

                    tag = "Adult" if adult else "Minor"
                    dist_m = cam_geo.get_distance_at_y(y2)
                    cv2.putText(debug_frame,
                                f"ID{tid}[{quality}] {h_cm:.0f}cm [{tag}] d={dist_m:.1f}m",
                                (x1, max(15,y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dc, 1)

                    # 키 측정 라인 (머리~발)
                    if head_y is not None and foot_y is not None:
                        cx = (x1+x2)//2
                        cv2.line(debug_frame, (cx, int(head_y)), (cx, int(foot_y)), (0,0,255), 2)

            # 범례
            ov = frame_vis.copy()
            cv2.rectangle(ov, (5,5),(220,150),(0,0,0),-1)
            cv2.addWeighted(ov, 0.6, frame_vis, 0.4, 0, frame_vis)
            for i,(ck,nm) in enumerate(CLASS_LABELS.items()):
                yp = 15+i*25
                cv2.rectangle(frame_vis,(10,yp),(25,yp+15),CLASS_COLORS[ck],-1)
                put_kr(frame_vis, nm, (32,yp-2), sz=14, c=(255,255,255))
            put_kr(frame_vis, f"기준: {args.adult_height_cm}cm",
                   (10,120), sz=12, c=(200,200,200))

            if writer: writer.write(frame_vis)
            if args.show:
                cv2.putText(debug_frame,
                    f"Cam: H={args.cam_height/1000:.1f}m Tilt={args.cam_tilt}deg "
                    f"FOV={args.cam_fov_v}deg | Adult>={args.adult_height_cm}cm",
                    (10,H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
                cv2.imshow("[Debug] Geometry + Height", debug_frame)
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

    # ── 히트맵 ──
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

    # ── CSV ──
    csv_path = os.path.join(args.output_dir, f"{source_name}_camgeo_tracking.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(["영상", "추적ID", "분류코드", "분류명",
                     "추정키_cm", "관절품질", "샘플수"])
        for tid in sorted(id_heights_mm.keys()):
            cls = id_final_class.get(tid, "미분류")
            cls_name = CLASS_LABELS.get(cls, "미분류")
            hs = id_heights_mm[tid]
            valid = [h for h, q in hs if q != "D" and h > 100]
            med_cm = np.median(valid)/10 if valid else 0
            best_q = min([q for _, q in hs], key=lambda x: "ABCD".index(x)) if hs else "D"
            w.writerow([source_name, tid, cls, cls_name,
                        f"{med_cm:.1f}", best_q, len(valid)])
    print(f"\n  → CSV: {csv_path}")

    # ── 결과 ──
    total_ids = len(set(id_heights_mm.keys()))
    classified = len(id_final_class)
    counts = Counter(id_final_class.values())

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    print(f"  총 고유 ID: {total_ids}명")
    print(f"  분류 확정:  {classified}명 ({classified/max(total_ids,1)*100:.0f}%)")
    print()
    for ck, nm in CLASS_LABELS.items():
        c = counts.get(ck, 0)
        print(f"  {nm}: {c}명 ({c/max(classified,1)*100:.1f}%)")

    all_cms = []
    for hs in id_heights_mm.values():
        valid = [h/10 for h, q in hs if q != "D" and h > 100]
        if valid: all_cms.append(np.median(valid))
    if all_cms:
        print(f"\n  추정 키 분포: {min(all_cms):.0f}cm ~ {max(all_cms):.0f}cm "
              f"(평균={np.mean(all_cms):.0f}cm, 중간값={np.median(all_cms):.0f}cm)")
    print("=" * 60)


if __name__ == "__main__":
    main()