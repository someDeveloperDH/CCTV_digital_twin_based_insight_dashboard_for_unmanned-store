"""
현재 simulationWorker.js의 구역 설정(ice1~ice6, beverage, snack1, snack2)에 맞는
학습용 batch_purchases CSV를 Python으로 생성한다.

사용법 (캡스톤 프로젝트 루트에서):
    python simul/server/generate_batch_data.py

출력:  model/data/batch_purchases_v2.csv  (120일, 9구역)
"""
import random
import math
import csv
from pathlib import Path

# ── simul 구역 설정 (simulationWorker.js 동일) ────────────────────────────────
ZONES = [
    {"id": "ice1",     "label": "아이스크림1", "preferred": ["adult_male"],
     "conversionRate": 0.50, "dwellRange": [3, 10],
     "items": [("수박바",700),("스크류바",600),("죠스바",700),("쮸쮸바",400),("아이스바",800)]},
    {"id": "ice2",     "label": "아이스크림2", "preferred": ["adult_male"],
     "conversionRate": 1.00, "dwellRange": [5, 15],
     "items": [("나뚜루파인트",3000),("하겐다즈",3500),("투게더",2500),("아이스세트",2800),("메로나4입",2400)]},
    {"id": "ice3",     "label": "아이스크림3", "preferred": ["adult_male","adult_female"],
     "conversionRate": 0.71, "dwellRange": [4, 12],
     "items": [("비비빅",1400),("부라보콘",1200),("구구콘",1200),("빵빠레",1400),("메로나",600)]},
    {"id": "ice4",     "label": "아이스크림4", "preferred": ["adult_male","minor_female"],
     "conversionRate": 1.00, "dwellRange": [8, 20],
     "items": [("투게더대",2000),("프리미엄하드",1800),("아이스케이크",2500),("대용량아이스",1500)]},
    {"id": "ice5",     "label": "아이스크림5", "preferred": ["adult_male","adult_female","minor_male","minor_female"],
     "conversionRate": 0.89, "dwellRange": [3, 12],
     "items": [("카나페",1200),("설레임",1200),("탱크보이",800),("스크류바",600),("수박바",600)]},
    {"id": "ice6",     "label": "아이스크림6", "preferred": ["adult_male","minor_male"],
     "conversionRate": 0.70, "dwellRange": [5, 20],
     "items": [("월드콘",1200),("빵빠레",1200),("구구콘",1200),("아이스샌드",1800),("하드보스",1200)]},
    {"id": "beverage", "label": "음료",         "preferred": ["adult_male","adult_female"],
     "conversionRate": 0.75, "dwellRange": [3, 10],
     "items": [("커피",1500),("아메리카노",2000),("카페라떼",2000),("포카리스웨트",1800),("파워에이드",1500)]},
    {"id": "snack1",   "label": "과자1",         "preferred": ["adult_male","minor_male"],
     "conversionRate": 1.00, "dwellRange": [3, 10],
     "items": [("포테이토칩",1500),("크래커",1200),("초코과자",1800),("새우과자",1500),("콘스낵",1200)]},
    {"id": "snack2",   "label": "과자2",         "preferred": ["minor_female","minor_male","adult_male"],
     "conversionRate": 0.43, "dwellRange": [5, 18],
     "items": [("바나나킥",500),("알감자",500),("짱구",500),("꼬깔콘",800),("사탕",300)]},
]

CLASS_KEYS     = ["adult_male", "adult_female", "minor_male", "minor_female"]
DAILY_VISITORS = 500      # 일 평균 방문자
OPERATING_HRS  = 12       # 10시~21시
START_HOUR     = 10
TOTAL_DAYS     = 120
SEED           = 42

# 시간대별 방문 비율 (10시~21시, simulationWorker의 hourlyProbs와 유사)
HOUR_WEIGHTS = [0.06, 0.08, 0.12, 0.13, 0.12, 0.09, 0.09, 0.08, 0.10, 0.07, 0.04, 0.02]

random.seed(SEED)

def gauss_random(mean, std):
    u = v = 0.0
    while True:
        u = random.random() * 2 - 1
        v = random.random() * 2 - 1
        s = u * u + v * v
        if 0 < s < 1:
            break
    return mean + std * u * math.sqrt(-2 * math.log(s) / s)

def daily_multiplier(day_idx):
    dow = day_idx % 7
    weekend_boost = 1.3 if dow >= 5 else 1.0
    noise = max(0.7, min(1.4, gauss_random(1.0, 0.1)))
    return weekend_boost * noise

def pick_class(ratios):
    total = sum(ratios.values())
    r = random.random() * total
    for cls, w in ratios.items():
        r -= w
        if r <= 0:
            return cls
    return CLASS_KEYS[0]

def build_queue(class_type):
    pref    = [z for z in ZONES if class_type in z["preferred"]]
    nonpref = [z for z in ZONES if class_type not in z["preferred"]]
    random.shuffle(pref)
    random.shuffle(nonpref)
    n_pref = 1 + int(random.random() * min(len(pref), 3))
    n_non  = min(len(nonpref), 1 + int(random.random() * 2))
    return pref[:n_pref] + nonpref[:n_non]

def simulate():
    rows = []
    agent_id = 0

    class_ratios = {"adult_male": 0.3, "adult_female": 0.3, "minor_male": 0.2, "minor_female": 0.2}

    for day in range(TOTAL_DAYS):
        mult     = daily_multiplier(day)
        n_visit  = max(50, int(gauss_random(DAILY_VISITORS * mult, DAILY_VISITORS * mult * 0.1)))

        for _ in range(n_visit):
            agent_id += 1
            class_type = pick_class(class_ratios)

            # 시간대 배정
            r = random.random()
            cum = 0.0
            hour_idx = 0
            for i, w in enumerate(HOUR_WEIGHTS):
                cum += w
                if r <= cum:
                    hour_idx = i
                    break
            actual_hour = START_HOUR + hour_idx

            # 구역 방문 & 구매
            queue = build_queue(class_type)
            for z in queue:
                dwell_lo, dwell_hi = z["dwellRange"]
                dwell = round(random.uniform(dwell_lo, dwell_hi), 1)

                if random.random() < z["conversionRate"]:
                    item_name, price = random.choice(z["items"])
                    rows.append({
                        "일차":       day + 1,
                        "구역ID":     z["id"],
                        "구역명":     z["label"],
                        "고객분류":   class_type,
                        "에이전트ID": agent_id,
                        "상품명":     item_name,
                        "가격":       price,
                        "체류시간":   dwell,
                        "시간대":     f"{actual_hour}시",
                    })

    return rows


def main():
    print("배치 데이터 생성 중... (120일, 9구역)")
    rows = simulate()

    out_path = Path(__file__).parent.parent.parent / "model" / "data" / "batch_purchases_v2.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["일차","구역ID","구역명","고객분류","에이전트ID","상품명","가격","체류시간","시간대"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"완료: {out_path}")
    print(f"총 행: {len(rows):,}  (일 평균 구매: {len(rows)/120:.0f}건)")

    # 간단 EDA
    import collections
    zone_cnt = collections.Counter(r["구역ID"] for r in rows)
    print("\n구역별 구매 건수:")
    for zid, cnt in sorted(zone_cnt.items(), key=lambda x: -x[1]):
        print(f"  {zid:12s}: {cnt:5,}")


if __name__ == "__main__":
    main()
