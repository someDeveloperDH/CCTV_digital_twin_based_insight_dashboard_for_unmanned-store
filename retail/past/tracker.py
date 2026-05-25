"""
tracker.py - YOLOv8n + ByteTrack 기반 사람 추적 모듈
"""

from ultralytics import YOLO
from collections import defaultdict


def run_tracking(video_path: str, model_name: str = "yolov8n.pt",
                 conf_threshold: float = 0.3) -> dict:
    """
    영상에서 사람을 검출하고 ByteTrack으로 추적

    Args:
        video_path: 입력 영상 경로
        model_name: YOLO 모델 (기본 yolov8n - CPU에 적합)
        conf_threshold: 검출 신뢰도 임계값

    Returns:
        dict: {frame_idx: [(track_id, x1, y1, x2, y2), ...]}
    """
    model = YOLO(model_name)

    tracking_results = defaultdict(list)

    # ByteTrack 추적 실행
    # classes=[0] → COCO 'person' 클래스만 검출
    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        classes=[0],            # person only
        conf=conf_threshold,
        iou=0.5,
        stream=True,            # 메모리 효율을 위해 스트리밍
        verbose=False,
    )

    frame_idx = 0
    for result in results:
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()     # (N, 4)
            track_ids = result.boxes.id.cpu().numpy()    # (N,)

            for box, tid in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                tracking_results[frame_idx].append(
                    (int(tid), float(x1), float(y1), float(x2), float(y2))
                )

        frame_idx += 1

    print(f"  → {frame_idx} 프레임 처리 완료")
    return dict(tracking_results)
