"""
utils.py - 유틸리티 함수 모음
"""

import os
import cv2
import numpy as np


def ensure_dir(path: str):
    """디렉토리가 없으면 생성"""
    os.makedirs(path, exist_ok=True)


def draw_bbox(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              label: str, color: tuple, thickness: int = 2):
    """바운딩 박스 + 라벨 그리기"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # 라벨 배경
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

    # 라벨 텍스트 (한글은 OpenCV에서 깨지므로 ID + 영문 약어 사용)
    # 실제 한글 표시가 필요하면 PIL을 사용해야 함
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_legend(frame: np.ndarray, class_labels: dict, class_colors: dict):
    """프레임 좌상단에 범례 표시"""
    # 영문 약어 매핑
    short_labels = {
        "young_male": "Young M",
        "young_female": "Young F",
        "old_male": "Older M",
        "old_female": "Older F",
    }

    x_start, y_start = 10, 20
    line_height = 25

    # 반투명 배경
    overlay = frame.copy()
    bg_h = len(class_labels) * line_height + 15
    cv2.rectangle(overlay, (5, 5), (150, bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for idx, (cls, kr_name) in enumerate(class_labels.items()):
        y = y_start + idx * line_height
        color = class_colors[cls]
        label = short_labels.get(cls, cls)

        # 색상 사각형
        cv2.rectangle(frame, (x_start, y - 10), (x_start + 15, y + 5), color, -1)

        # 텍스트
        cv2.putText(frame, label, (x_start + 22, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
