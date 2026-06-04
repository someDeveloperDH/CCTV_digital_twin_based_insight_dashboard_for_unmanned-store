import cv2
import time
import torch
import urllib.request
import sys
from pathlib import Path
from ultralytics import YOLO

MERGED_DIR  = Path(__file__).parent / "merged"
OUT_DIR     = Path(__file__).parent / "mosaic"
MODEL_PATH  = Path(__file__).parent / "yolov8n-face.pt"

MODEL_URL = (
    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt"
)

CONF_THRESHOLD = 0.5   # 얼굴 검출 신뢰도 임계값
MOSAIC_SCALE   = 0.05  # 모자이크 강도 (낮을수록 강함)
PADDING        = 0.15  # 얼굴 박스 여백 비율
BATCH_SIZE     = 32    # GPU 병렬 처리 프레임 수 (최적: 32에서 속도 포화)
USE_FP16       = True  # FP16 추론 (CUDA)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def download_model():
    if MODEL_PATH.exists():
        return
    print(f"YOLO face 모델 다운로드 중...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("모델 다운로드 완료")


def apply_mosaic(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    small = cv2.resize(roi, (max(1, int(w * MOSAIC_SCALE)), max(1, int(h * MOSAIC_SCALE))))
    frame[y:y+h, x:x+w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def expand_box(x1, y1, x2, y2, img_w, img_h, pad=PADDING):
    pw = int((x2 - x1) * pad)
    ph = int((y2 - y1) * pad)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(img_w, x2 + pw)
    y2 = min(img_h, y2 + ph)
    return x1, y1, x2, y2


def process_video(video_path: Path, model: YOLO):
    """Returns metrics dict: {frames, faces, fps, skipped}"""
    out_path = OUT_DIR / video_path.name
    if out_path.exists():
        print(f"  [SKIP] 이미 존재: {out_path.name}")
        return {"frames": 0, "faces": 0, "fps": 0.0, "skipped": True}
    t0 = time.time()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] 열기 실패: {video_path.name}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx       = 0
    face_count_total = 0
    batch_frames    = []
    batch_indices   = []

    def process_batch(frames, indices):
        nonlocal face_count_total
        results = model(frames, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
        for frame, result in zip(frames, results):
            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box[:4])
                    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, width, height)
                    apply_mosaic(frame, x1, y1, x2 - x1, y2 - y1)
                    face_count_total += 1
            writer.write(frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            if batch_frames:
                process_batch(batch_frames, batch_indices)
            break

        batch_frames.append(frame)
        batch_indices.append(frame_idx)
        frame_idx += 1

        if len(batch_frames) == BATCH_SIZE:
            process_batch(batch_frames, batch_indices)
            batch_frames.clear()
            batch_indices.clear()

        if frame_idx % 300 == 0:
            pct = frame_idx / total * 100 if total else 0
            print(f"    {frame_idx}/{total} ({pct:.1f}%) 처리 중...")

    cap.release()
    writer.release()
    elapsed = max(time.time() - t0, 0.001)
    fps_val = round(frame_idx / elapsed, 1)
    print(f"  [OK] {out_path.name} | {frame_idx}프레임 | 얼굴 {face_count_total}건 | {fps_val}fps")
    return {"frames": frame_idx, "faces": face_count_total, "fps": fps_val, "skipped": False}


def main():
    download_model()
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})\n")
    model = YOLO(str(MODEL_PATH))
    if USE_FP16 and DEVICE == "cuda":
        model.model.half()

    videos = sorted(MERGED_DIR.glob("*.mp4"))
    if not videos:
        print("merged 폴더에 mp4 파일이 없습니다.")
        sys.exit(1)

    print(f"총 {len(videos)}개 영상 처리 시작\n")
    all_metrics = []
    t_all = time.time()
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")
        all_metrics.append(process_video(video, model))

    total_frames = sum(m["frames"] for m in all_metrics if m)
    total_faces  = sum(m["faces"]  for m in all_metrics if m)
    elapsed_all  = max(time.time() - t_all, 0.001)
    print(f"\n전체 완료 | 총 {total_frames}프레임 | 얼굴 {total_faces}건 | "
          f"전체평균 {total_frames/elapsed_all:.1f}fps")
    return all_metrics


if __name__ == "__main__":
    main()
