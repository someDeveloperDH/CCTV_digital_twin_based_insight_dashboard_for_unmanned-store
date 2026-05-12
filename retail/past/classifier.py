"""
classifier.py - DeepFace 기반 성별/연령 분류 모듈 (개선 버전)

CCTV 환경 최적화:
- 상체 영역을 확대 crop하여 얼굴 검출률 향상
- 다양한 crop 전략 (상체 1/3, 상체 1/2, 전체)을 순차 시도
- 작은 crop은 업스케일하여 DeepFace 검출률 향상
- ID당 여러 프레임에서 시도 후 다수결(majority vote)로 최종 분류
"""

import cv2
import numpy as np
from collections import Counter


class AgeGenderClassifier:
    def __init__(self, detector_backend: str = "opencv"):
        self.detector_backend = detector_backend
        self._deepface = None
        self.vote_buffer = {}  # {track_id: [(age, gender), ...]}

    def _lazy_import(self):
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace
            print("  [INFO] DeepFace 모델 로딩중 (최초 1회)...")
            try:
                dummy = np.zeros((100, 100, 3), dtype=np.uint8)
                self._deepface.analyze(
                    dummy,
                    actions=["age", "gender"],
                    detector_backend=self.detector_backend,
                    enforce_detection=False,
                    silent=True,
                )
            except Exception:
                pass
            print("  [INFO] DeepFace 모델 로딩 완료")

    def _try_analyze(self, img: np.ndarray) -> tuple | None:
        try:
            results = self._deepface.analyze(
                img,
                actions=["age", "gender"],
                detector_backend=self.detector_backend,
                enforce_detection=False,
                silent=True,
            )
            if isinstance(results, list):
                result = results[0]
            else:
                result = results
            age = int(result["age"])
            gender = result["dominant_gender"]
            return (age, gender)
        except Exception:
            return None

    def _upscale(self, img: np.ndarray, target_min: int = 120) -> np.ndarray:
        """작은 이미지를 업스케일"""
        h, w = img.shape[:2]
        scale = max(1, target_min // max(min(h, w), 1))
        if scale > 1:
            return cv2.resize(img, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR)
        return img

    def predict(self, person_crop: np.ndarray) -> tuple | None:
        """
        여러 crop 전략을 순차 시도하여 검출률 향상
        """
        self._lazy_import()

        h, w = person_crop.shape[:2]
        if h < 30 or w < 20:
            return None

        # 전략 1: 상체 1/3 (얼굴 포함 가능성 높음) → 업스케일
        upper_third = person_crop[0:h // 3, :]
        if upper_third.shape[0] >= 20:
            result = self._try_analyze(self._upscale(upper_third))
            if result is not None:
                return result

        # 전략 2: 상체 1/2 → 업스케일
        upper_half = person_crop[0:h // 2, :]
        if upper_half.shape[0] >= 20:
            result = self._try_analyze(self._upscale(upper_half))
            if result is not None:
                return result

        # 전략 3: 전체 crop → 업스케일
        result = self._try_analyze(self._upscale(person_crop, target_min=150))
        if result is not None:
            return result

        return None

    def add_vote(self, track_id: int, age: int, gender: str):
        """분류 결과를 투표 버퍼에 추가"""
        if track_id not in self.vote_buffer:
            self.vote_buffer[track_id] = []
        self.vote_buffer[track_id].append((age, gender))

    def get_majority_vote(self, track_id: int, min_votes: int = 1) -> tuple | None:
        """다수결로 최종 분류 결과 반환"""
        if track_id not in self.vote_buffer:
            return None
        votes = self.vote_buffer[track_id]
        if len(votes) < min_votes:
            return None

        genders = [g for _, g in votes]
        majority_gender = Counter(genders).most_common(1)[0][0]
        avg_age = int(np.mean([a for a, _ in votes]))
        return (avg_age, majority_gender)