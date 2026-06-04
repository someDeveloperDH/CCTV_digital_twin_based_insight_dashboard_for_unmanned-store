"""simul purchases JSON → batch_purchases.csv 형식 DataFrame 변환."""
import pandas as pd
from typing import List, Dict, Any


def purchases_to_df(purchases: List[Dict[str, Any]], operating_hours: int) -> pd.DataFrame:
    """
    simul purchases 항목 리스트를 dataset.py가 기대하는 CSV 형식 DataFrame으로 변환한다.

    simul 입력:  { agentId, classType, zoneId, zoneLabel, item, price, dwellTime, day, hour }
                  hour: 0-base 운영시간 인덱스 (0 ~ operatingHours-1)
                  day:  0-base

    출력 컬럼:   일차, 구역ID, 구역명, 고객분류, 에이전트ID, 상품명, 가격, 체류시간, 시간대
                  시간대: "10시" 등 절대 시간 문자열
                  일차:   1-base
    """
    start_hour = max(6, 22 - operating_hours)

    rows = []
    for p in purchases:
        rows.append({
            "일차":     p["day"] + 1,
            "구역ID":   p["zoneId"],
            "구역명":   p["zoneLabel"],
            "고객분류": p["classType"],
            "에이전트ID": p["agentId"],
            "상품명":   p.get("item", ""),
            "가격":     float(p["price"]),
            "체류시간": float(p["dwellTime"]),
            "시간대":   f"{p['hour'] + start_hour}시",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["일차", "구역ID", "구역명", "고객분류", "에이전트ID", "상품명", "가격", "체류시간", "시간대"]
    )
