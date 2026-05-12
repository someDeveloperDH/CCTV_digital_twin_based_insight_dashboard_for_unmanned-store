"""
test_pose.py - YOLOv8n-pose 기반 키 측정 테스트

포즈 estimation으로 관절 좌표 추출 → 머리~발목 pixel 거리로 키 측정
+ ByteTrack 추적 + 2점 선형 보정 + MiVOLO 성별 분류

YOLOv8n-pose 관절 17개 (COCO format):
  0:코  1:왼눈  2:오눈  3:왼귀  4:오귀
  5:왼어깨  6:오어깨  7:왼팔꿈치  8:오팔꿈치
  9:왼손목  10:오손목  11:왼엉덩이  12:오엉덩이
  13:왼무릎  14:오무릎  15:왼발목  16:오발목

키 측정: 머리(코 or 눈) y좌표 ~ 발목(15,16 평균) y좌표 차이

사용법:
    python test_pose.py --source data/test1.mp4 --show
    python test_pose.py --source data/test1.mp4 --save-video --show
    python test_pose.py --source data/test1.mp4 --save-video --show --height-threshold 90
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
from collections import Counter, defaultdict
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ══════════════════════════════════════════════
# MiVOLO v2 (성별만 사용)
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
# 4클래스 정의
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

# 관절 연결선 (skeleton)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # 얼굴
    (5, 6),                                   # 어깨
    (5, 7), (7, 9),                           # 왼팔
    (6, 8), (8, 10),                          # 오른팔
    (5, 11), (6, 12),                         # 몸통
    (11, 12),                                 # 엉덩이
    (11, 13), (13, 15),                       # 왼다리
    (12, 14), (14, 16),                       # 오른다리
]


# ══════════════════════════════════════════════
# 포즈 기반 키 측정
# ══════════════════════════════════════════════
def estimate_height_from_keypoints(keypoints: np.ndarray, conf_threshold: float = 0.3) -> float | None:
    """
    관절 좌표에서 키(pixel) 측정

    전략:
      1. 코(0) ~ 발목(15,16) 거리 (가장 정확)
      2. 눈(1,2) ~ 발목 거리 (코가 안 보일 때)
      3. 어깨(5,6) ~ 발목 거리 × 보정계수 (상체만 보일 때)
      4. bbox 높이로 폴백 (관절이 거의 안 보일 때)

    Args:
        keypoints: (17, 3) 배열 [x, y, confidence]
        conf_threshold: 관절 신뢰도 임계값

    Returns:
        키(pixel) 또는 None (측정 불가)
    """
    kp = keypoints  # (17, 3)

    # 발목 y좌표 (왼발목15, 오른발목16 중 유효한 것)
    ankle_y = None
    if kp[15][2] > conf_threshold and kp[16][2] > conf_threshold:
        ankle_y = max(kp[15][1], kp[16][1])  # 더 아래쪽
    elif kp[15][2] > conf_threshold:
        ankle_y = kp[15][1]
    elif kp[16][2] > conf_threshold:
        ankle_y = kp[16][1]

    # 무릎이라도 있으면 발목 추정 (무릎~발목 비율 약 1.15배)
    if ankle_y is None:
        knee_y = None
        if kp[13][2] > conf_threshold and kp[14][2] > conf_threshold:
            knee_y = max(kp[13][1], kp[14][1])
        elif kp[13][2] > conf_threshold:
            knee_y = kp[13][1]
        elif kp[14][2] > conf_threshold:
            knee_y = kp[14][1]

        if knee_y is not None:
            # 엉덩이~무릎 거리로 무릎~발목 추정
            hip_y = None
            if kp[11][2] > conf_threshold and kp[12][2] > conf_threshold:
                hip_y = max(kp[11][1], kp[12][1])
            elif kp[11][2] > conf_threshold:
                hip_y = kp[11][1]
            elif kp[12][2] > conf_threshold:
                hip_y = kp[12][1]

            if hip_y is not None:
                shin_est = (knee_y - hip_y) * 1.1  # 정강이 ≈ 허벅지 × 1.1
                ankle_y = knee_y + shin_est

    if ankle_y is None:
        return None

    # 머리 y좌표 (코 → 눈 → 귀 순서로 시도)
    head_y = None
    if kp[0][2] > conf_threshold:
        head_y = kp[0][1]
    elif kp[1][2] > conf_threshold and kp[2][2] > conf_threshold:
        head_y = min(kp[1][1], kp[2][1])
    elif kp[1][2] > conf_threshold:
        head_y = kp[1][1]
    elif kp[2][2] > conf_threshold:
        head_y = kp[2][1]
    elif kp[3][2] > conf_threshold or kp[4][2] > conf_threshold:
        ear_ys = [kp[i][1] for i in [3, 4] if kp[i][2] > conf_threshold]
        head_y = min(ear_ys)

    if head_y is not None:
        height = ankle_y - head_y
        # 코에서 정수리까지 약 15% 추가
        height *= 1.15
        return max(0, height)

    # 어깨만 보이는 경우: 어깨~발목 × 1.25 (어깨 위 부분 보정)
    shoulder_y = None
    if kp[5][2] > conf_threshold and kp[6][2] > conf_threshold:
        shoulder_y = min(kp[5][1], kp[6][1])
    elif kp[5][2] > conf_threshold:
        shoulder_y = kp[5][1]
    elif kp[6][2] > conf_threshold:
        shoulder_y = kp[6][1]

    if shoulder_y is not None:
        height = (ankle_y - shoulder_y) * 1.35  # 어깨~발 = 전체 키의 약 74%
        return max(0, height)

    return None


def get_keypoint_quality(keypoints: np.ndarray, conf_threshold: float = 0.3) -> str:
    """관절 검출 품질 등급"""
    kp = keypoints
    n_valid = sum(1 for i in range(17) if kp[i][2] > conf_threshold)

    has_head = any(kp[i][2] > conf_threshold for i in [0, 1, 2])
    has_ankle = any(kp[i][2] > conf_threshold for i in [15, 16])
    has_knee = any(kp[i][2] > conf_threshold for i in [13, 14])

    if has_head and has_ankle and n_valid >= 10:
        return "A"  # 머리~발 완전
    elif has_head and has_knee:
        return "B"  # 머리~무릎 (발목 추정)
    elif has_ankle or has_knee:
        return "C"  # 부분적
    else:
        return "D"  # 불량


# ══════════════════════════════════════════════
# 2점 선형 보정
# ══════════════════════════════════════════════
class TwoPointCalibrator:
    """
    영상 내 2명의 성인 키 데이터로 y좌표별 보정 계수 자동 계산

    원리: 같은 키의 사람이 가까이(하단)에서 180px, 멀리(상단)에서 60px이면
          y좌표별 보정계수를 선형 보간으로 계산
    """

    def __init__(self, frame_height: int):
        self.H = frame_height
        self.data_points = []  # [(y_center, raw_height), ...]
        self.slope = 0.0
        self.intercept = 0.0
        self.calibrated = False
        self.ref_height = None  # 하단 기준 높이

    def add_sample(self, y_center: float, raw_height: float, quality: str = "A"):
        """키 데이터 수집 (품질 나쁜 데이터는 제외)"""
        # quality D(관절 실패) 또는 키가 너무 작은 데이터는 보정을 오염시킴
        if quality == "D":
            return
        if raw_height < 30:
            return

        self.data_points.append((y_center, raw_height))

        # 충분한 데이터가 모이면 자동 캘리브레이션 (주기적으로 재계산)
        if len(self.data_points) >= 20 and not self.calibrated:
            self._calibrate()
        elif self.calibrated and len(self.data_points) % 100 == 0:
            self._calibrate()  # 주기적 재보정

    def _calibrate(self):
        """선형 회귀로 y→키 관계 학습"""
        ys = np.array([p[0] for p in self.data_points])
        hs = np.array([p[1] for p in self.data_points])

        if np.std(ys) < 10:
            return  # y 분산이 너무 작으면 보정 의미 없음

        # h = slope * y + intercept
        self.slope, self.intercept = np.polyfit(ys, hs, 1)

        # 기준 높이 = 하단(가까운 곳)에서의 예상 키
        self.ref_height = self.slope * self.H + self.intercept

        # ref_height가 비정상이면 중간값 사용
        if self.ref_height <= 0:
            self.ref_height = float(np.median(hs))

        self.calibrated = True

        # 자동 임계값 추천 (중간값의 60% = 미성년 기준선)
        median_corrected = float(np.median(hs))
        self.auto_threshold = median_corrected * 0.6

        print(f"  [보정] 자동 캘리브레이션 완료 ({len(self.data_points)}샘플): "
              f"h = {self.slope:.3f}*y + {self.intercept:.1f}")
        print(f"         기준키={self.ref_height:.0f}px, "
              f"원본중간값={median_corrected:.0f}px, "
              f"추천임계값={self.auto_threshold:.0f}px")

    def correct_height(self, raw_height: float, y_center: float) -> float:
        """원근 보정된 키 반환 — 상단(멀리)에 있는 사람의 키를 키움"""
        if not self.calibrated:
            return raw_height

        # 이 y좌표에서 예상되는 평균 키
        expected_h = self.slope * y_center + self.intercept
        if expected_h > 0 and self.ref_height > 0:
            # 보정계수: 하단 기준 대비 현재 위치의 축소 비율
            scale = self.ref_height / expected_h
            # scale > 1이면 상단(멀리) → 키를 키움
            # scale ≈ 1이면 하단(가까이) → 거의 그대로
            corrected = raw_height * scale
            return max(corrected, 0)
        return raw_height

    def get_auto_threshold(self) -> float | None:
        """자동 추천 임계값 반환"""
        if self.calibrated and hasattr(self, 'auto_threshold'):
            return self.auto_threshold
        return None

    def get_info(self) -> str:
        if self.calibrated:
            thr_str = f", auto_thr={self.auto_threshold:.0f}px" if hasattr(self, 'auto_threshold') else ""
            return f"Auto-calibrated ({len(self.data_points)} samples, ref={self.ref_height:.0f}px{thr_str})"
        return f"Collecting... ({len(self.data_points)}/20 samples)"


# ══════════════════════════════════════════════
# MiVOLO 성별 추정
# ══════════════════════════════════════════════
def predict_gender(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    pad_x = int((x2 - x1) * 0.05)
    pad_y = int((y2 - y1) * 0.05)
    bx1, by1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    bx2, by2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    body_crop = frame[by1:by2, bx1:bx2]
    if body_crop.size == 0:
        return None

    face_h = (y2 - y1) // 3
    face_crop = frame[max(0, y1):min(h, y1 + face_h), max(0, x1):min(w, x2)]

    try:
        body_input = IMAGE_PROCESSOR(images=[body_crop])["pixel_values"]
        body_input = body_input.to(dtype=MIVOLO_MODEL.dtype, device=MIVOLO_MODEL.device)

        if face_crop.size > 0 and face_crop.shape[0] >= 10 and face_crop.shape[1] >= 10:
            faces_input = IMAGE_PROCESSOR(images=[face_crop])["pixel_values"]
            faces_input = faces_input.to(dtype=MIVOLO_MODEL.dtype, device=MIVOLO_MODEL.device)
        else:
            faces_input = torch.zeros_like(body_input)

        with torch.no_grad():
            output = MIVOLO_MODEL(faces_input=faces_input, body_input=body_input)

        gender_id = output.gender_class_idx[0].item()
        gender = CONFIG.gender_id2label[gender_id]
        gender_prob = output.gender_probs[0].item() * 100
        return {"gender": gender, "gender_prob": round(gender_prob, 1)}
    except Exception:
        return None


# ══════════════════════════════════════════════
# 성별 다수결 투표
# ══════════════════════════════════════════════
class GenderVoter:
    def __init__(self, min_votes=3, min_conf=55.0):
        self.votes = defaultdict(list)
        self.finalized = {}
        self.min_votes = min_votes
        self.min_conf = min_conf

    def add_vote(self, tid, gender, prob):
        if tid in self.finalized:
            return
        if prob >= self.min_conf:
            self.votes[tid].append((gender, prob))
        if len(self.votes[tid]) >= self.min_votes:
            genders = [v[0] for v in self.votes[tid]]
            self.finalized[tid] = Counter(genders).most_common(1)[0][0]
            avg_prob = np.mean([v[1] for v in self.votes[tid]])
            g_kr = "남성" if self.finalized[tid] == "male" else "여성"
            print(f"  ✓ ID {tid} 성별 확정: {g_kr} "
                  f"(투표 {len(self.votes[tid])}회, 신뢰도={avg_prob:.0f}%)")

    def get_gender(self, tid):
        return self.finalized.get(tid, None)

    def get_tentative_gender(self, tid):
        if tid in self.finalized:
            return self.finalized[tid]
        votes = self.votes.get(tid, [])
        if votes:
            return Counter([v[0] for v in votes]).most_common(1)[0][0]
        return None

    def needs_vote(self, tid):
        return tid not in self.finalized


# ══════════════════════════════════════════════
# 한글 렌더링
# ══════════════════════════════════════════════
_FONT_CACHE = {}

def get_font(size=18):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for fp in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc",
               "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
               "/System/Library/Fonts/AppleGothic.ttf"]:
        if os.path.exists(fp):
            try:
                _FONT_CACHE[size] = ImageFont.truetype(fp, size)
                return _FONT_CACHE[size]
            except Exception:
                continue
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]

def put_kr(frame, text, pos, size=18, color=(255,255,255), bg=None):
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = get_font(size)
    rc = (color[2], color[1], color[0])
    x, y = pos
    if bg:
        rb = (bg[2], bg[1], bg[0])
        bb = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bb[0]-2, bb[1]-2, bb[2]+2, bb[3]+2], fill=rb)
    draw.text((x, y), text, font=font, fill=rc)
    np.copyto(frame, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))


# ══════════════════════════════════════════════
# 스켈레톤 그리기
# ══════════════════════════════════════════════
def draw_skeleton(frame, keypoints, color=(0, 255, 255), conf_threshold=0.3):
    """관절 좌표로 스켈레톤 그리기"""
    kp = keypoints
    # 관절 점
    for i in range(17):
        if kp[i][2] > conf_threshold:
            x, y = int(kp[i][0]), int(kp[i][1])
            cv2.circle(frame, (x, y), 3, color, -1)

    # 연결선
    for (i, j) in SKELETON_CONNECTIONS:
        if kp[i][2] > conf_threshold and kp[j][2] > conf_threshold:
            pt1 = (int(kp[i][0]), int(kp[i][1]))
            pt2 = (int(kp[j][0]), int(kp[j][1]))
            cv2.line(frame, pt1, pt2, color, 1)


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Pose 기반 키 측정 + MiVOLO 성별 분류")
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
                        help="보정된 키 기준 성년/미성년 임계값 (px)")
    parser.add_argument("--auto-calibrate", action="store_true", default=True,
                        help="2점 자동 원근 보정 (기본 켜짐)")
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
    print("Pose Estimation + MiVOLO 성별 분류")
    print("=" * 60)
    print(f"입력: {args.source} ({W}x{H}, {fps:.0f}fps, {total}프레임)")
    print(f"키 임계값: {args.height_threshold}px (보정 후)")
    print(f"자동 원근 보정: {'ON' if args.auto_calibrate else 'OFF'}")
    print("=" * 60)

    # 모델 로드
    print("\n[로드] YOLOv8n-pose...")
    yolo = YOLO("../yolov8n-pose.pt")

    # 모듈 초기화
    calibrator = TwoPointCalibrator(H) if args.auto_calibrate else None
    gender_voter = GenderVoter(min_votes=args.min_votes, min_conf=args.min_conf)
    attempt_count = defaultdict(int)

    # 비디오 라이터
    writer = None
    if args.save_video:
        out_path = os.path.join(args.output_dir, f"pose_test_{source_name}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print("[처리] 프레임 처리중...\n")
    cap = cv2.VideoCapture(args.source)
    frame_idx = 0

    id_heights = defaultdict(list)       # {tid: [(raw_h, corrected_h, y_center, quality), ...]}
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
        keypoints_data = results[0].keypoints

        if boxes is not None and boxes.id is not None and keypoints_data is not None:
            xyxys = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            all_kps = keypoints_data.data.cpu().numpy()  # (N, 17, 3)

            for idx, (box, tid) in enumerate(zip(xyxys, ids)):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                kps = all_kps[idx]  # (17, 3)

                # 포즈 기반 키 측정
                raw_h = estimate_height_from_keypoints(kps)
                bbox_h = y2 - y1
                y_center = (y1 + y2) / 2
                quality = get_keypoint_quality(kps)

                # 관절이 안 보이면 bbox 높이로 폴백
                if raw_h is None:
                    raw_h = float(bbox_h)
                    quality = "D"

                # 2점 자동 보정
                corrected_h = raw_h
                if calibrator:
                    calibrator.add_sample(y_center, raw_h, quality)
                    corrected_h = calibrator.correct_height(raw_h, y_center)

                current_tracks.append((tid, x1, y1, x2, y2, kps, raw_h, corrected_h, quality))
                id_heights[tid].append((raw_h, corrected_h, y_center, quality))

                # ── 성별 분류 (MiVOLO) ──
                if frame_idx % args.skip_frames != 0:
                    continue
                if not gender_voter.needs_vote(tid):
                    continue
                if attempt_count[tid] >= args.max_attempts:
                    continue
                if bbox_h < 50:
                    continue

                attempt_count[tid] += 1
                result = predict_gender(frame, x1, y1, x2, y2)
                if result:
                    gender_voter.add_vote(tid, result["gender"], result["gender_prob"])

        # ── 임계값 결정 (자동 or 수동) ──
        effective_threshold = args.height_threshold
        if calibrator:
            auto_thr = calibrator.get_auto_threshold()
            if auto_thr is not None:
                effective_threshold = auto_thr

        # ── 최종 4클래스 결정 ──
        for track in current_tracks:
            tid = track[0]
            if tid in id_final_class:
                continue

            gender = gender_voter.get_gender(tid)
            if gender is None:
                continue

            heights = id_heights.get(tid, [])
            if len(heights) < 3:
                continue

            # 보정된 키 중간값 (quality D 제외)
            corrected_list = [h[1] for h in heights if h[3] != "D" and h[1] > 10]
            if not corrected_list:
                corrected_list = [h[1] for h in heights if h[1] > 10]
            if not corrected_list:
                continue

            median_corrected = np.median(corrected_list)

            is_adult = median_corrected >= effective_threshold

            if is_adult and gender == "male":
                id_final_class[tid] = "adult_male"
            elif is_adult and gender == "female":
                id_final_class[tid] = "adult_female"
            elif not is_adult and gender == "male":
                id_final_class[tid] = "minor_male"
            else:
                id_final_class[tid] = "minor_female"

            median_raw = np.median([h[0] for h in heights if h[0] > 10])
            cls_name = CLASS_LABELS[id_final_class[tid]]
            print(f"  ★ ID {tid} 최종: {cls_name} "
                  f"(원본={median_raw:.0f}px, 보정={median_corrected:.0f}px, "
                  f"임계값={effective_threshold:.0f}px, "
                  f"성년={'O' if is_adult else 'X'}, 성별={gender})")

        # ── 시각화 ──
        if args.save_video or args.show:
            frame_vis = frame.copy()

            # 디버그 프레임 (스켈레톤 + 키 측정)
            debug_frame = frame.copy() if args.show else None

            for track in current_tracks:
                tid, x1, y1, x2, y2, kps, raw_h, corrected_h, quality = track
                cls = id_final_class.get(tid, None)

                # ── 메인 영상: 분류 결과 ──
                if cls:
                    color = CLASS_COLORS[cls]
                    label = f"ID{tid} {CLASS_LABELS[cls]}"
                else:
                    tent_g = gender_voter.get_tentative_gender(tid)
                    color = UNCLASSIFIED_COLOR
                    g_str = "남?" if tent_g == "male" else ("여?" if tent_g == "female" else "?")
                    label = f"ID{tid} {g_str}"

                cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, 2)
                put_kr(frame_vis, label, (x1, max(0, y1 - 22)),
                       size=15, color=(255, 255, 255), bg=color)

                # ── 디버그 영상: 스켈레톤 + 키 수치 ──
                if debug_frame is not None:
                    is_ad = corrected_h >= effective_threshold
                    dbg_color = (0, 255, 0) if is_ad else (0, 255, 255)

                    draw_skeleton(debug_frame, kps, color=dbg_color)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), dbg_color, 1)

                    tag = "Adult" if is_ad else "Minor"
                    cv2.putText(debug_frame,
                                f"ID{tid} [{quality}] {raw_h:.0f}->{corrected_h:.0f}px [{tag}]",
                                (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, dbg_color, 1)

            # 범례
            overlay = frame_vis.copy()
            cv2.rectangle(overlay, (5, 5), (200, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame_vis, 0.4, 0, frame_vis)
            for i, (ck, name) in enumerate(CLASS_LABELS.items()):
                y_pos = 15 + i * 25
                cv2.rectangle(frame_vis, (10, y_pos), (25, y_pos + 15), CLASS_COLORS[ck], -1)
                put_kr(frame_vis, name, (32, y_pos - 2), size=14, color=(255, 255, 255))

            if writer:
                writer.write(frame_vis)
            if args.show:
                # 디버그 하단 정보
                calib_info = calibrator.get_info() if calibrator else "OFF"
                cv2.putText(debug_frame,
                            f"Threshold: {effective_threshold:.0f}px | Calibration: {calib_info}",
                            (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

                cv2.imshow("[Debug] Pose + Height", debug_frame)
                cv2.imshow("[Final] Classification", frame_vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_idx += 1
        if frame_idx % 50 == 0:
            calib_str = calibrator.get_info() if calibrator else ""
            print(f"  프레임 {frame_idx}/{total} | "
                  f"추적 {len(current_tracks)}명 | 확정 {len(id_final_class)}명 | {calib_str}")

    cap.release()
    if writer:
        writer.release()
        print(f"\n  → 결과 영상: {out_path}")
    if args.show:
        cv2.destroyAllWindows()

    # ── 최종 결과 ──
    total_ids = len(set(id_heights.keys()))
    classified = len(id_final_class)
    counts = Counter(id_final_class.values())

    print("\n" + "=" * 60)
    print("최종 분류 결과")
    print("=" * 60)
    print(f"  총 고유 ID: {total_ids}명")
    print(f"  분류 확정:  {classified}명 ({classified/max(total_ids,1)*100:.0f}%)")
    if calibrator:
        print(f"  원근 보정:  {calibrator.get_info()}")
    print()
    for ck, name in CLASS_LABELS.items():
        c = counts.get(ck, 0)
        pct = (c / max(classified, 1)) * 100
        print(f"  {name}: {c}명 ({pct:.1f}%)")

    # 키 분포
    all_raw = []
    all_corrected = []
    for hlist in id_heights.values():
        if hlist:
            raws = [h[0] for h in hlist]
            corrs = [h[1] for h in hlist]
            all_raw.append(np.median(raws))
            all_corrected.append(np.median(corrs))
    if all_raw:
        print(f"\n  원본 키: min={min(all_raw):.0f} max={max(all_raw):.0f} avg={np.mean(all_raw):.0f}px")
        print(f"  보정 키: min={min(all_corrected):.0f} max={max(all_corrected):.0f} avg={np.mean(all_corrected):.0f}px")
    print("=" * 60)


if __name__ == "__main__":
    main()