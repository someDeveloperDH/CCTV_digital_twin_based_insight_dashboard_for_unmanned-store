"""
heatmap_generator.py - 클래스별 히트맵 생성 모듈

각 클래스(남녀노소)별로 좌표를 누적하고
Gaussian blur를 적용하여 히트맵 이미지 생성
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없는 환경 대응
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os


class HeatmapGenerator:
    def __init__(self, width: int, height: int,
                 resolution_scale: float = 1.0,
                 class_labels: dict = None,
                 sigma: int = 30):
        """
        Args:
            width: 원본 영상 너비
            height: 원본 영상 높이
            resolution_scale: 히트맵 해상도 배율
            class_labels: {class_key: 한글이름} 딕셔너리
            sigma: Gaussian blur 커널 크기 (클수록 부드러운 히트맵)
        """
        self.orig_w = width
        self.orig_h = height
        self.scale = resolution_scale
        self.w = int(width * resolution_scale)
        self.h = int(height * resolution_scale)
        self.sigma = sigma
        self.class_labels = class_labels or {}

        # 클래스별 누적 맵
        self.accum_maps = {}
        for cls in self.class_labels:
            self.accum_maps[cls] = np.zeros((self.h, self.w), dtype=np.float32)

        # 한글 폰트 설정 시도
        self._setup_korean_font()

    def _setup_korean_font(self):
        """시스템에서 한글 폰트 탐색"""
        korean_font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for fpath in korean_font_paths:
            if os.path.exists(fpath):
                font_manager.fontManager.addfont(fpath)
                prop = font_manager.FontProperties(fname=fpath)
                plt.rcParams['font.family'] = prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                return

        # 한글 폰트 없으면 영문 fallback
        print("  [WARN] 한글 폰트를 찾을 수 없어 영문으로 표시합니다.")
        self._use_english_labels()

    def _use_english_labels(self):
        """한글 폰트 없을 때 영문 라벨 사용"""
        self._english_labels = {
            "young_male": "Young Male",
            "young_female": "Young Female",
            "old_male": "Older Male",
            "old_female": "Older Female",
        }

    def _get_label(self, cls: str) -> str:
        if hasattr(self, '_english_labels'):
            return self._english_labels.get(cls, cls)
        return self.class_labels.get(cls, cls)

    def add_point(self, cls: str, x: float, y: float):
        """좌표를 누적 맵에 추가"""
        sx = int(x * self.scale)
        sy = int(y * self.scale)

        if 0 <= sx < self.w and 0 <= sy < self.h:
            self.accum_maps[cls][sy, sx] += 1

    def _make_heatmap(self, accum: np.ndarray) -> np.ndarray:
        """누적 맵에 Gaussian blur 적용하여 히트맵 생성"""
        if accum.max() == 0:
            return np.zeros((self.h, self.w), dtype=np.float32)

        # Gaussian blur로 스무딩
        kernel_size = self.sigma * 4 + 1  # 항상 홀수
        blurred = cv2.GaussianBlur(accum, (kernel_size, kernel_size), self.sigma)

        # 정규화
        if blurred.max() > 0:
            blurred = blurred / blurred.max()

        return blurred

    def save_individual_heatmap(self, cls: str, save_path: str):
        """특정 클래스의 히트맵을 저장"""
        heatmap = self._make_heatmap(self.accum_maps[cls])

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(heatmap, cmap='jet', interpolation='bilinear',
                  extent=[0, self.orig_w, self.orig_h, 0],
                  vmin=0, vmax=1, alpha=0.8)
        ax.set_title(self._get_label(cls), fontsize=16, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Density')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_combined_heatmap(self, save_path: str):
        """전체 클래스 통합 히트맵"""
        total_accum = np.zeros((self.h, self.w), dtype=np.float32)
        for cls in self.accum_maps:
            total_accum += self.accum_maps[cls]

        heatmap = self._make_heatmap(total_accum)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(heatmap, cmap='jet', interpolation='bilinear',
                  extent=[0, self.orig_w, self.orig_h, 0],
                  vmin=0, vmax=1, alpha=0.8)

        title = "Combined Heatmap" if hasattr(self, '_english_labels') else "전체 통합 히트맵"
        ax.set_title(title, fontsize=16, fontweight='bold')

        cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Density')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_comparison_heatmap(self, save_path: str):
        """2x2 클래스 비교 히트맵"""
        classes = list(self.class_labels.keys())

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, cls in enumerate(classes):
            heatmap = self._make_heatmap(self.accum_maps[cls])
            ax = axes[idx]

            im = ax.imshow(heatmap, cmap='jet', interpolation='bilinear',
                           extent=[0, self.orig_w, self.orig_h, 0],
                           vmin=0, vmax=1, alpha=0.8)
            ax.set_title(self._get_label(cls), fontsize=14, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        title = "Heatmap by Group" if hasattr(self, '_english_labels') else "남녀노소별 히트맵 비교"
        fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
