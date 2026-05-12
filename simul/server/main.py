"""KPI 예측 FastAPI 서버 — http://localhost:8000"""
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .predict import load_models, run_inference
from .converter import purchases_to_df


# ── 요청/응답 스키마 ──────────────────────────────────────────────────────────

class PurchaseItem(BaseModel):
    agentId:   int
    classType: str
    zoneId:    str
    zoneLabel: str
    item:      Optional[str] = ""
    price:     float
    dwellTime: float
    day:       int
    hour:      int


class ZoneStatItem(BaseModel):
    visitors: int
    buyers:   int
    label:    Optional[str] = ""


class HourlyStatItem(BaseModel):
    hour:     int
    visitors: int
    buyers:   int


class PredictRequest(BaseModel):
    operatingHours: int
    purchases:      List[PurchaseItem]
    zoneStats:      Dict[str, ZoneStatItem]
    hourlyStats:    List[HourlyStatItem]


# ── 앱 초기화 ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[KPI 서버] 모델 로딩 중...")
    load_models()
    print("[KPI 서버] 준비 완료 - http://localhost:8000")
    yield
    print("[KPI 서버] 종료")


app = FastAPI(title="KPI Prediction Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict-kpi")
def predict_kpi(req: PredictRequest):
    if not req.purchases:
        raise HTTPException(400, detail="purchases 데이터가 비어 있습니다.")

    df = purchases_to_df(
        [p.model_dump() for p in req.purchases],
        req.operatingHours,
    )

    zone_stats_dict = {
        k: v.model_dump() for k, v in req.zoneStats.items()
    }
    hourly_stats_list = [h.model_dump() for h in req.hourlyStats]

    # known zones 0개이면 400 반환
    known_zones = set(zone_stats_dict.keys())
    if not known_zones:
        raise HTTPException(
            400,
            detail="zoneStats가 비어 있습니다. 시뮬레이션 데이터가 충분한지 확인해 주세요.",
        )

    try:
        result = run_inference(df, zone_stats_dict, hourly_stats_list, req.operatingHours)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        raise HTTPException(500, detail=f"Inference 오류: {e}")

    if not result["zoneKPI"]:
        raise HTTPException(
            400,
            detail=(
                "학습 데이터와 일치하는 구역이 없습니다. "
                "같은 매장 유형으로 시뮬레이션 후 다시 시도해 주세요. "
                f"(알 수 없는 구역: {result['unknownZones']})"
            ),
        )

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simul.server.main:app", host="0.0.0.0", port=8000, reload=False)
