"""
visit_data.json + zones.json → 3개 CSV 출력 + 터미널 요약
출력:
  ice_batch_purchases.csv  - 개인별 구역/상품 상세
  ice_class_kpi.csv        - 고객 분류별 KPI
  ice_zone_kpi.csv         - 구역별 KPI
실행: python export.py
"""
import json
import pandas as pd
from pathlib import Path
from tabulate import tabulate
from collections import defaultdict

BASE_DIR    = Path(__file__).parent
RESULT_FILE = BASE_DIR / "visit_data.json"
ZONES_FILE  = BASE_DIR / "zones.json"

OUT_PURCHASES = BASE_DIR / "ice_batch_purchases.csv"
OUT_CLASS_KPI = BASE_DIR / "ice_class_kpi.csv"
OUT_ZONE_KPI  = BASE_DIR / "ice_zone_kpi.csv"

DAY_NUMBER = 1   # 분석 대상 일차 (10일치 = 1일차)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_data():
    with open(RESULT_FILE, encoding="utf-8") as f:
        visits = json.load(f)
    zones = []
    if ZONES_FILE.exists():
        with open(ZONES_FILE, encoding="utf-8") as f:
            zones = json.load(f)
    return visits, zones


# ── 고객 분류 ─────────────────────────────────────────────────────────────────

def get_classification(info):
    gender   = info.get("gender")
    is_adult = info.get("is_adult")

    # is_adult가 문자열로 저장된 경우 변환
    if isinstance(is_adult, str):
        is_adult = is_adult.lower() == "true"

    if gender == "M" and is_adult is True:
        return "adult_male",   "성인 남성"
    if gender == "F" and is_adult is True:
        return "adult_female", "성인 여성"
    if gender == "M" and is_adult is False:
        return "minor_male",   "미성인 남성"
    if gender == "F" and is_adult is False:
        return "minor_female", "미성인 여성"
    if gender == "M":
        return "adult_male",   "성인 남성"
    if gender == "F":
        return "adult_female", "성인 여성"
    return "unknown", "미확인"


# ── CSV 1: ice_batch_purchases ────────────────────────────────────────────────

def build_purchases(visits, zones):
    """
    에이전트ID, 고객분류, 고객분류명, 구역ID, 구역명, 상품명,
    개수, 가격, 체류시간, 일차, 시간대
    """
    zone_map = {z["name"]: z for z in zones}   # name → zone dict (id, price, products 포함)
    rows = []

    for vid, info in visits.items():
        cls_id, cls_name = get_classification(info)
        zone_dwell = info.get("zones", {})
        purchased  = info.get("purchased", False)
        timeband   = info.get("timeband", "-")

        visited_zones = [(zname, secs) for zname, secs in zone_dwell.items() if secs > 0]

        if not visited_zones:
            # 구역 방문 없음
            rows.append({
                "에이전트ID": int(vid[1:]),
                "고객분류":   cls_id,
                "고객분류명": cls_name,
                "구역ID":     "none",
                "구역명":     "none",
                "상품명":     "none",
                "개수":       1,
                "가격":       0,
                "체류시간":   0,
                "일차":       DAY_NUMBER,
                "시간대":     timeband,
            })
            continue

        for zname, dwell_sec in visited_zones:
            zone_info = zone_map.get(zname, {})
            products  = zone_info.get("products", [])
            zone_id   = zname.replace(" ", "_")

            if purchased and products:
                unit_price = zone_info.get("price", 0)
                price_map  = zone_info.get("price_map", {})
                for product in products:
                    price = price_map.get(product, unit_price)
                    rows.append({
                        "에이전트ID": int(vid[1:]),
                        "고객분류":   cls_id,
                        "고객분류명": cls_name,
                        "구역ID":     zone_id,
                        "구역명":     zname,
                        "상품명":     product,
                        "개수":       1,
                        "가격":       price,
                        "체류시간":   round(dwell_sec),
                        "일차":       DAY_NUMBER,
                        "시간대":     timeband,
                    })
            else:
                rows.append({
                    "에이전트ID": int(vid[1:]),
                    "고객분류":   cls_id,
                    "고객분류명": cls_name,
                    "구역ID":     zone_id,
                    "구역명":     zname,
                    "상품명":     "none",
                    "개수":       1,
                    "가격":       0,
                    "체류시간":   round(dwell_sec),
                    "일차":       DAY_NUMBER,
                    "시간대":     timeband,
                })

    return pd.DataFrame(rows)


# ── CSV 2: ice_class_kpi ──────────────────────────────────────────────────────

def calc_visit_sales(info, zones):
    """방문 구역 상품 가격 합산 (구매한 경우에만)"""
    if not info.get("purchased"):
        return 0
    zone_map = {z["name"]: z for z in zones}
    total = 0
    for zname in info.get("zones", {}):
        z = zone_map.get(zname, {})
        price_map = z.get("price_map", {})
        unit_price = z.get("price", 0)
        products   = z.get("products", [])
        for p in products:
            total += price_map.get(p, unit_price)
    return total


def build_class_kpi(visits, zones):
    """
    분류, 분류명, 방문객수, 구매고객수, 구매전환율(%), 총매출, 평균구매액
    """
    stats = defaultdict(lambda: {"방문객수": 0, "구매고객수": 0, "총매출": 0})

    for vid, info in visits.items():
        cls_id, cls_name = get_classification(info)
        stats[cls_id]["분류명"]   = cls_name
        stats[cls_id]["방문객수"] += 1
        if info.get("purchased"):
            stats[cls_id]["구매고객수"] += 1
            stats[cls_id]["총매출"]     += calc_visit_sales(info, zones)

    rows = []
    for cls_id, d in sorted(stats.items()):
        visitors  = d["방문객수"]
        purchasers = d["구매고객수"]
        total_sales = d["총매출"]
        conv  = round(purchasers / visitors * 100, 1) if visitors else 0
        avg   = round(total_sales / purchasers) if purchasers else 0
        rows.append({
            "분류":          cls_id,
            "분류명":        d["분류명"],
            "방문객수":      visitors,
            "구매고객수":    purchasers,
            "구매전환율(%)": conv,
            "총매출":        total_sales,
            "평균구매액":    avg,
        })

    return pd.DataFrame(rows)


# ── CSV 3: ice_zone_kpi ───────────────────────────────────────────────────────

def build_zone_kpi(visits, zones):
    """
    구역, 방문객수, 구매고객수, 구매전환율(%), 평균체류시간(초),
    총매출, 구매고객비율(%), 체류대비효율, 비효율지수
    """
    stats = defaultdict(lambda: {
        "방문객수": 0, "구매고객수": 0,
        "총체류": 0.0, "총매출": 0
    })

    for vid, info in visits.items():
        zone_dwell = info.get("zones", {})
        purchased  = info.get("purchased", False)
        for zname, secs in zone_dwell.items():
            if secs <= 0:
                continue
            stats[zname]["방문객수"]  += 1
            stats[zname]["총체류"]    += secs
            if purchased:
                stats[zname]["구매고객수"] += 1
                stats[zname]["총매출"]     += calc_visit_sales(info, zones)

    total_zone_purchasers = sum(d["구매고객수"] for d in stats.values()) or 1

    rows = []
    for zname in [z["name"] for z in zones]:
        d = stats.get(zname)
        if not d:
            continue
        visitors    = d["방문객수"]
        purchasers  = d["구매고객수"]
        total_dwell = d["총체류"]
        total_sales = d["총매출"]
        avg_dwell   = round(total_dwell / visitors, 1) if visitors else 0
        conv        = round(purchasers / visitors * 100, 1) if visitors else 0
        buy_ratio   = round(purchasers / total_zone_purchasers * 100, 1)

        # 체류대비효율 = 총매출 / 총체류시간
        efficiency  = round(total_sales / total_dwell, 4) if total_dwell > 0 else 0
        # 비효율지수 = 비구매 전환율(%)
        inefficiency = round((1 - conv / 100) * 100, 1)

        rows.append({
            "구역":            zname,
            "방문객수":        visitors,
            "구매고객수":      purchasers,
            "구매전환율(%)":   conv,
            "평균체류시간(초)": avg_dwell,
            "총매출":          total_sales,
            "구매고객비율(%)": buy_ratio,
            "체류대비효율":    efficiency,
            "비효율지수":      inefficiency,
        })

    return pd.DataFrame(rows)


# ── 터미널 출력 ───────────────────────────────────────────────────────────────

def print_section(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def print_summary(df_purchases, df_class, df_zone):
    print_section("[ 테이블 1 ] 개인별 구역/상품 상세 (상위 20행)")
    print(tabulate(df_purchases.head(20), headers="keys",
                   tablefmt="rounded_outline", showindex=False))
    print(f"  총 {len(df_purchases)}행  /  에이전트 {df_purchases['에이전트ID'].nunique()}명")

    print_section("[ 테이블 2 ] 고객 분류별 KPI")
    print(tabulate(df_class, headers="keys",
                   tablefmt="rounded_outline", showindex=False))

    print_section("[ 테이블 3 ] 구역별 KPI")
    print(tabulate(df_zone, headers="keys",
                   tablefmt="rounded_outline", showindex=False))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not RESULT_FILE.exists():
        print("visit_data.json 없음 → analyze.py 먼저 실행하세요.")
        return

    visits, zones = load_data()
    print(f"방문자 {len(visits)}명 로드 완료\n")

    df_purchases = build_purchases(visits, zones)
    df_class     = build_class_kpi(visits, zones)
    df_zone      = build_zone_kpi(visits, zones)

    # 터미널 출력
    print_summary(df_purchases, df_class, df_zone)

    # CSV 저장
    df_purchases.to_csv(OUT_PURCHASES, index=False, encoding="utf-8-sig")
    df_class.to_csv(OUT_CLASS_KPI,     index=False, encoding="utf-8-sig")
    df_zone.to_csv(OUT_ZONE_KPI,       index=False, encoding="utf-8-sig")

    print(f"\nCSV 저장 완료:")
    print(f"  {OUT_PURCHASES.name}")
    print(f"  {OUT_CLASS_KPI.name}")
    print(f"  {OUT_ZONE_KPI.name}")


if __name__ == "__main__":
    main()
