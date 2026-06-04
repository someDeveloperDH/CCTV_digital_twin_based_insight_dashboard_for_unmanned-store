"""
스캐너 구역 기반 상품 감지 + ProductMatcher

흐름:
  1. ch1에서 "스캐너" zone 폴리곤 감시
  2. 배경 차분으로 물건이 구역 안으로 들어옴 감지
  3. ProductMatcher: 크롭 → product_db/embeddings.npz 와 코사인 유사도
  4. zones.json["products"]에서 zone_name / price 조회
  5. (product_name, zone_name, price, similarity) 반환
"""
import cv2
import json
import timm
import torch
import numpy as np
import torchvision.transforms as T
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_FILE  = BASE_DIR / "product_db" / "embeddings.npz"
ZONES_FILE = BASE_DIR / "zones.json"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# 감지 파라미터
MIN_CHANGE_AREA  = 500   # 픽셀 수 — 노이즈 제거
PERSIST_FRAMES   = 6     # N프레임 연속 변화 = 물건이 놓임
DEBOUNCE_FRAMES  = 60    # 중복 감지 방지 (약 2초 @ 30fps)
MATCH_THRESHOLD  = 0.55  # 코사인 유사도 임계값 (낮추면 관대, 높이면 엄격)


# ── 상품 매처 ─────────────────────────────────────────────────────────────────

class ProductMatcher:
    """
    product_db/embeddings.npz 로드 후 코사인 유사도 1-NN 매칭.
    zones.json에서 상품명 → 구역명/가격 조회.
    """

    def __init__(self):
        self._zone_lookup = self._load_zone_lookup()
        self._names       = []
        self._embeddings  = None
        self._model       = None
        self._tf          = None
        self._ready       = False
        self._load_db()

    def _load_zone_lookup(self):
        """상품명 → {zone_name, zone_id, price}"""
        if not ZONES_FILE.exists():
            return {}
        with open(ZONES_FILE, encoding="utf-8") as f:
            zones = json.load(f)
        lookup = {}
        for z in zones:
            for product in z.get("products", []):
                lookup[product] = {
                    "zone_name": z["name"],
                    "zone_id":   z["id"],
                    "price":     z.get("price", 0),
                }
        return lookup

    def _load_db(self):
        if not DB_FILE.exists():
            print(f"  [ProductDB] {DB_FILE} 없음 → build_product_db.py 먼저 실행")
            return
        data = np.load(DB_FILE, allow_pickle=True)
        self._names      = list(data["names"])
        self._embeddings = data["embeddings"].astype(np.float32)  # (N, D)
        self._load_model()
        self._ready = True
        print(f"  [ProductDB] {len(self._names)}개 상품 로드: {self._names[:5]}{'...' if len(self._names)>5 else ''}")

    def _load_model(self):
        self._model = timm.create_model(
            "mobilenetv3_small_100", pretrained=True, num_classes=0
        ).to(DEVICE).eval()
        self._tf = T.Compose([
            T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def _extract(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return None
        rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t    = self._tf(rgb).unsqueeze(0).to(DEVICE)
        feat = self._model(t).cpu().numpy()[0].astype(np.float32)
        n    = np.linalg.norm(feat)
        return feat / n if n > 0 else feat

    def match(self, crop_bgr):
        """
        Returns: {"product": name, "zone_name": str, "zone_id": str,
                  "price": int, "similarity": float}
        or None (매칭 실패 / DB 없음)
        """
        if not self._ready:
            return None
        query = self._extract(crop_bgr)
        if query is None:
            return None

        # 코사인 유사도 (embeddings는 이미 정규화됨)
        sims = self._embeddings @ query          # (N,)
        best_idx  = int(sims.argmax())
        best_sim  = float(sims[best_idx])
        best_name = self._names[best_idx]

        if best_sim < MATCH_THRESHOLD:
            return None

        info = self._zone_lookup.get(best_name, {})
        return {
            "product":    best_name,
            "zone_name":  info.get("zone_name", "미분류"),
            "zone_id":    info.get("zone_id",   "unknown"),
            "price":      info.get("price",     0),
            "similarity": round(best_sim, 3),
        }

    def top_k(self, crop_bgr, k=3):
        """디버그용: 상위 k개 후보 반환"""
        if not self._ready:
            return []
        query = self._extract(crop_bgr)
        if query is None:
            return []
        sims   = self._embeddings @ query
        idxs   = sims.argsort()[::-1][:k]
        return [(self._names[i], round(float(sims[i]), 3)) for i in idxs]


# ── 스캐너 구역 감시 ──────────────────────────────────────────────────────────

class ScannerDetector:
    """
    "스캐너" zone 폴리곤을 감시하다가 물건이 놓이면 ProductMatcher로 식별.
    analyze.py의 process_camera()에서 매 프레임 update() 호출.
    """

    def __init__(self, scanner_polygon, matcher: ProductMatcher):
        self.polygon    = np.array(scanner_polygon, dtype=np.int32)
        self.matcher    = matcher
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=35, detectShadows=False)
        self._persist   = 0
        self._best_crop = None
        self._last_fi   = -DEBOUNCE_FRAMES

    # ── 내부 ──────────────────────────────────────────────────────────────────

    def _roi_mask(self, shape):
        m = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(m, [self.polygon], 255)
        return m

    def _foreground(self, frame, mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, mask)
        fg   = self.subtractor.apply(gray)
        fg   = cv2.bitwise_and(fg, mask)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg   = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        fg   = cv2.morphologyEx(fg, cv2.MORPH_DILATE, k)
        return fg

    def _crop(self, frame, fg):
        pts = cv2.findNonZero(fg)
        if pts is None:
            return None
        x, y, w, h = cv2.boundingRect(pts)
        if w < 20 or h < 20:
            return None
        return frame[y:y+h, x:x+w].copy()

    # ── 공개 인터페이스 ───────────────────────────────────────────────────────

    def update(self, frame, fi):
        """
        매 프레임 호출.
        상품 감지 시 dict 반환, 아니면 None.
        """
        if fi - self._last_fi < DEBOUNCE_FRAMES:
            return None

        mask = self._roi_mask(frame.shape)
        fg   = self._foreground(frame, mask)
        area = cv2.countNonZero(fg)

        if area >= MIN_CHANGE_AREA:
            self._persist += 1
            crop = self._crop(frame, fg)
            # 선명도가 더 높은 크롭을 유지 (흔들림 최소화)
            if crop is not None:
                if self._best_crop is None:
                    self._best_crop = crop
                else:
                    if cv2.Laplacian(crop, cv2.CV_64F).var() > \
                       cv2.Laplacian(self._best_crop, cv2.CV_64F).var():
                        self._best_crop = crop
        else:
            self._persist   = 0
            self._best_crop = None

        # PERSIST_FRAMES 프레임 연속 감지 → 매칭 실행
        if self._persist >= PERSIST_FRAMES and self._best_crop is not None:
            result = self.matcher.match(self._best_crop)
            # 디버그 후보 출력
            top3 = self.matcher.top_k(self._best_crop, k=3)
            print(f"    [스캐너] 상위3: {top3}")
            self._persist   = 0
            self._best_crop = None
            if result:
                self._last_fi = fi
                print(f"    [스캐너] 감지: {result['product']}  "
                      f"구역={result['zone_name']}  "
                      f"가격={result['price']}원  "
                      f"유사도={result['similarity']}")
                return result
        return None

    def draw_debug(self, frame, last_result=None):
        """시각화: 스캐너 구역 + 감지 결과"""
        cv2.polylines(frame, [self.polygon], True, (0, 255, 255), 2)
        if last_result:
            x = self.polygon[:, 0].min()
            y = self.polygon[:, 1].min() - 10
            txt = f"{last_result['product']} / {last_result['zone_name']} / {last_result['price']}원"
            cv2.putText(frame, txt, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame
