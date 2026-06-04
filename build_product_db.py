"""
상품 이미지 DB 구축 스크립트

사용법:
  1. product_db/ 폴더에 상품 이미지 저장
     파일명 = 상품명  (예: 더위사냥.jpg, 구구콘.jpg, 월드콘.png)
  2. python build_product_db.py 실행
  3. product_db/embeddings.npz 생성됨 (분석 시 자동 로드)

권장 촬영 방법:
  - 계산대 스캐너 구역과 동일한 각도에서 촬영
  - 밝은 곳에서 배경 없이 상품만 촬영
  - 여러 각도 버전은 파일명에 _1, _2 등 추가 (예: 더위사냥_1.jpg)
    → 같은 상품의 여러 임베딩이 모두 DB에 저장됨
"""
import cv2
import json
import timm
import torch
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from tabulate import tabulate

BASE_DIR    = Path(__file__).parent
DB_DIR      = BASE_DIR / "product_db"
ZONES_FILE  = BASE_DIR / "zones.json"
OUTPUT_FILE = DB_DIR / "embeddings.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── 임베딩 모델 ───────────────────────────────────────────────────────────────

def load_model():
    model = timm.create_model(
        "mobilenetv3_small_100", pretrained=True, num_classes=0
    ).to(DEVICE).eval()
    tf = T.Compose([
        T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf


@torch.no_grad()
def extract(img_bgr, model, tf):
    if img_bgr is None or img_bgr.size == 0:
        return None
    rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t    = tf(rgb).unsqueeze(0).to(DEVICE)
    feat = model(t).cpu().numpy()[0]
    n    = np.linalg.norm(feat)
    return feat / n if n > 0 else feat


# ── zones.json 상품 조회 ──────────────────────────────────────────────────────

def load_zone_lookup():
    """상품명 → (zone_name, price) 매핑 테이블"""
    if not ZONES_FILE.exists():
        return {}
    with open(ZONES_FILE, encoding="utf-8") as f:
        zones = json.load(f)
    lookup = {}
    for zone in zones:
        for product in zone.get("products", []):
            lookup[product] = {
                "zone_name": zone["name"],
                "zone_id":   zone["id"],
                "price":     zone.get("price", 0),
            }
    return lookup


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"DB 폴더: {DB_DIR}\n")

    img_paths = sorted(p for p in DB_DIR.iterdir()
                       if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        print("product_db/ 폴더에 이미지가 없습니다.")
        print("상품 사진을 찍어 product_db/<상품명>.jpg 형식으로 저장하세요.")
        return

    model, tf      = load_model()
    zone_lookup    = load_zone_lookup()
    names, embeds  = [], []
    rows           = []

    for path in img_paths:
        img  = cv2.imread(str(path))
        emb  = extract(img, model, tf)
        if emb is None:
            rows.append([path.name, "❌ 읽기 실패", "-", "-"])
            continue

        # 파일명에서 상품명 추출 (_1, _2 등 접미사 제거)
        stem         = path.stem
        product_name = stem.rsplit("_", 1)[0] if stem[-1].isdigit() else stem

        info     = zone_lookup.get(product_name, {})
        zone_str = f"{info.get('zone_name','?')}  ({info.get('price',0)}원)" \
                   if info else "⚠️ zones.json에 없음"

        names.append(product_name)
        embeds.append(emb)
        rows.append([path.name, product_name, zone_str, f"{len(emb)}dim ✅"])

    print(tabulate(rows, headers=["파일", "상품명", "구역/가격", "임베딩"],
                   tablefmt="rounded_outline"))

    if not embeds:
        print("\n임베딩 생성 실패")
        return

    np.savez(OUTPUT_FILE,
             names=np.array(names),
             embeddings=np.array(embeds))

    print(f"\n✅ 저장 완료: {OUTPUT_FILE}")
    print(f"   상품 수: {len(names)}개")

    # zones.json에 없는 상품 경고
    missing = [n for n in set(names) if n not in zone_lookup]
    if missing:
        print(f"\n⚠️  zones.json \"products\"에 없는 상품 → 구역/가격 조회 불가:")
        for m in missing:
            print(f"   - {m}")
        print("   zones.json의 해당 구역 'products' 리스트에 추가하세요.")


if __name__ == "__main__":
    main()
