"""KPI 예측 엔진 — Transformer + Crossformer 체크포인트 로드 후 inference."""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.model import PurchaseGridTransformer  # noqa: E402 (after sys.path insert)

# ── Crossformer 모델 클래스 (kpi_crossformer_local.py에서 복사 — import 불가) ────


class TwoStageAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_router, dropout=0.1):
        super().__init__()
        self.time_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_t     = nn.LayerNorm(d_model)
        self.router     = nn.Parameter(torch.randn(1, n_router, d_model))
        self.gather_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.broad_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_r     = nn.LayerNorm(d_model)
        self.norm_d     = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        B, L, N, d = x.shape
        xt     = x.permute(0, 2, 1, 3).reshape(B * N, L, d)
        xt2, _ = self.time_attn(xt, xt, xt)
        xt     = self.norm_t(xt + self.drop(xt2))
        x      = xt.reshape(B, N, L, d).permute(0, 2, 1, 3)

        xd     = x.reshape(B * L, N, d)
        r      = self.router.expand(B * L, -1, -1)
        r2, _  = self.gather_attn(r, xd, xd)
        r      = self.norm_r(r + self.drop(r2))
        xd2, _ = self.broad_attn(xd, r, r)
        xd     = self.norm_d(xd + self.drop(xd2))
        x      = xd.reshape(B, L, N, d)

        x = self.norm_f(x + self.drop(self.ffn(x)))
        return x


class KPICrossformer(nn.Module):
    def __init__(self, n_zones, n_hours, n_classes, n_features,
                 d_model=64, n_heads=4, n_layers=2, n_router=8,
                 dropout=0.1, horizon=1, max_window=64):
        super().__init__()
        self.n_zones, self.n_hours, self.n_classes = n_zones, n_hours, n_classes
        self.n_tokens = n_zones * n_hours * n_classes

        self.input_proj = nn.Linear(n_features, d_model)
        self.zone_emb   = nn.Embedding(n_zones,    d_model)
        self.hour_emb   = nn.Embedding(n_hours,    d_model)
        self.cls_emb    = nn.Embedding(n_classes,  d_model)
        self.pos_emb    = nn.Embedding(max_window, d_model)
        self.dow_emb    = nn.Embedding(7,          d_model)

        self.blocks = nn.ModuleList([
            TwoStageAttentionLayer(d_model, n_heads, n_router, dropout)
            for _ in range(n_layers)
        ])
        self.purchase_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))
        self.dwell_head    = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x, dow):
        B, L, N, _ = x.shape
        Z, T, C    = self.n_zones, self.n_hours, self.n_classes

        h = self.input_proj(x)

        zone_ids  = torch.arange(Z, device=x.device).repeat_interleave(T * C)
        hour_ids  = torch.arange(T, device=x.device).repeat(Z).repeat_interleave(C)
        cls_ids   = torch.arange(C, device=x.device).repeat(Z * T)
        token_emb = self.zone_emb(zone_ids) + self.hour_emb(hour_ids) + self.cls_emb(cls_ids)
        h = h + token_emb.view(1, 1, N, -1)
        h = h + self.pos_emb(torch.arange(L, device=x.device).view(1, L, 1))
        h = h + self.dow_emb(dow).unsqueeze(2)

        for block in self.blocks:
            h = block(h)

        last     = h[:, -1, :, :]
        purchase = self.purchase_head(last).permute(0, 2, 1)

        last_zt  = last.reshape(B, Z, T, C, -1).mean(dim=3).reshape(B, Z * T, -1)
        dwell    = self.dwell_head(last_zt).permute(0, 2, 1)

        return purchase, dwell


# ── 체크포인트 경로 ───────────────────────────────────────────────────────────

TF_CKPT_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best.pt"
CF_CKPT_PATH = PROJECT_ROOT / "checkpoints_crossformer" / "best_kpi.pt"


# ── 싱글톤 모델 상태 ─────────────────────────────────────────────────────────

class _S:
    tf_ckpt: Optional[dict] = None
    tf_model: Optional[PurchaseGridTransformer] = None
    cf_ckpt: Optional[dict] = None
    cf_model: Optional[KPICrossformer] = None
    device = torch.device("cpu")


def load_models() -> None:
    """서버 startup 시 1회 호출 — 두 모델을 메모리에 로드."""
    if not TF_CKPT_PATH.exists():
        raise FileNotFoundError(f"Transformer 체크포인트 없음: {TF_CKPT_PATH}")
    if not CF_CKPT_PATH.exists():
        raise FileNotFoundError(f"Crossformer 체크포인트 없음: {CF_CKPT_PATH}")

    # Transformer
    _S.tf_ckpt = torch.load(TF_CKPT_PATH, map_location=_S.device, weights_only=False)
    mc = _S.tf_ckpt["cfg_model"]
    W  = _S.tf_ckpt.get("window_size", 7)
    H  = _S.tf_ckpt.get("horizon", 1)
    _S.tf_model = PurchaseGridTransformer(
        n_zones=_S.tf_ckpt["n_zones"],
        n_hours=_S.tf_ckpt["n_hours"],
        n_features=_S.tf_ckpt["n_features"],
        d_model=mc["d_model"], n_heads=mc["n_heads"], n_layers=mc["n_layers"],
        dropout=0.0, horizon=H, max_window=max(W, 64),
    ).to(_S.device)
    _S.tf_model.load_state_dict(_S.tf_ckpt["model"])
    _S.tf_model.eval()

    # Crossformer
    _S.cf_ckpt = torch.load(CF_CKPT_PATH, map_location=_S.device, weights_only=False)
    mc_cf = _S.cf_ckpt["cfg_model"]
    dc_cf = _S.cf_ckpt["cfg_data"]
    _S.cf_model = KPICrossformer(
        n_zones=_S.cf_ckpt["n_zones"],
        n_hours=_S.cf_ckpt["n_hours"],
        n_classes=_S.cf_ckpt["n_classes"],
        n_features=4,
        d_model=mc_cf["d_model"], n_heads=mc_cf["n_heads"],
        n_layers=mc_cf["n_layers"], n_router=mc_cf["n_router"],
        dropout=0.0, horizon=dc_cf.get("horizon", 1),
        max_window=max(dc_cf.get("window_size", 5), 64),
    ).to(_S.device)
    _S.cf_model.load_state_dict(_S.cf_ckpt["model"])
    _S.cf_model.eval()

    print(f"  [Transformer] {_S.tf_ckpt['n_zones']}구역 × {_S.tf_ckpt['n_hours']}시간대  loaded")
    print(f"  [Crossformer] {_S.cf_ckpt['n_zones']}×{_S.cf_ckpt['n_hours']}×{_S.cf_ckpt['n_classes']} tokens  loaded")


# ── Grid 빌더 ────────────────────────────────────────────────────────────────

def _normalize_tf(grid: np.ndarray) -> np.ndarray:
    stats   = _S.tf_ckpt["stats"]
    means   = stats["means"]
    stds    = stats["stds"]
    log_idx = set(stats.get("log_indices", [0, 3]))
    skip    = set(stats.get("skip_indices", [4, 5, 6, 7]))
    out     = grid.copy()
    for i in range(grid.shape[-1]):
        if i in skip:
            continue
        if i in log_idx:
            out[..., i] = np.log1p(out[..., i])
        out[..., i] = (out[..., i] - means[i]) / stds[i]
    return out.astype(np.float32)


def _normalize_cf(grid: np.ndarray) -> np.ndarray:
    stats   = _S.cf_ckpt["stats"]
    means   = stats["means"]
    stds    = stats["stds"]
    log_idx = set(stats.get("log_idx", [0, 3]))
    out     = grid.copy()
    for i in range(grid.shape[-1]):
        if i in log_idx:
            out[..., i] = np.log1p(out[..., i])
        out[..., i] = (out[..., i] - means[i]) / stds[i]
    return out.astype(np.float32)


def _build_tf_grid(df: pd.DataFrame):
    """[D, Z, T, F=8] normalized grid + days list + unknown_zones list."""
    zone2idx = _S.tf_ckpt["zone2idx"]
    hour2idx = _S.tf_ckpt["hour2idx"]
    cls2idx  = _S.tf_ckpt["cls2idx"]
    Z = _S.tf_ckpt["n_zones"]
    T = _S.tf_ckpt["n_hours"]
    F = _S.tf_ckpt["n_features"]

    df = df.copy()
    df["hour_int"] = df["시간대"].str.replace("시", "").astype(int)

    all_sim_zones  = set(df["구역ID"].unique())
    known_zones    = set(zone2idx.keys())
    unknown_zones  = sorted(all_sim_zones - known_zones)

    df = df[df["구역ID"].isin(known_zones) & df["hour_int"].isin(hour2idx)]

    days_present = sorted(df["일차"].unique())
    D = len(days_present)
    if D == 0:
        return None, [], unknown_zones
    day2seq = {d: i for i, d in enumerate(days_present)}

    grid = np.zeros((D, Z, T, F), np.float32)

    for (day, zone, hour), grp in df.groupby(["일차", "구역ID", "hour_int"]):
        d  = day2seq.get(day)
        zi = zone2idx.get(zone)
        ti = hour2idx.get(hour)
        if d is None or zi is None or ti is None:
            continue
        total = len(grp)
        grid[d, zi, ti, 0] = total
        grid[d, zi, ti, 1] = grp["체류시간"].mean()
        grid[d, zi, ti, 2] = grp["가격"].mean()
        grid[d, zi, ti, 3] = grp["에이전트ID"].nunique()
        for cls, grp_cls in grp.groupby("고객분류"):
            ci = cls2idx.get(cls)
            if ci is not None and ci < 4:
                grid[d, zi, ti, 4 + ci] = len(grp_cls) / total

    return _normalize_tf(grid), days_present, unknown_zones


def _build_cf_grid(df: pd.DataFrame):
    """[D, Z, T, C, F=4] normalized grid + days list."""
    zone2idx = _S.cf_ckpt["zone2idx"]
    hour2idx = _S.cf_ckpt["hour2idx"]
    cls2idx  = _S.cf_ckpt["cls2idx"]
    Z = _S.cf_ckpt["n_zones"]
    T = _S.cf_ckpt["n_hours"]
    C = _S.cf_ckpt["n_classes"]
    F = 4

    df = df.copy()
    df["hour_int"] = df["시간대"].str.replace("시", "").astype(int)
    df = df[df["구역ID"].isin(zone2idx) & df["hour_int"].isin(hour2idx) & df["고객분류"].isin(cls2idx)]

    days_present = sorted(df["일차"].unique())
    D = len(days_present)
    if D == 0:
        return None, []
    day2seq = {d: i for i, d in enumerate(days_present)}

    grid = np.zeros((D, Z, T, C, F), np.float32)

    for (day, zone, hour, cls), grp in df.groupby(["일차", "구역ID", "hour_int", "고객분류"]):
        d  = day2seq.get(day)
        zi = zone2idx.get(zone)
        ti = hour2idx.get(hour)
        ci = cls2idx.get(cls)
        if any(v is None for v in [d, zi, ti, ci]):
            continue
        grid[d, zi, ti, ci, 0] = len(grp)
        grid[d, zi, ti, ci, 1] = grp["체류시간"].mean()
        grid[d, zi, ti, ci, 2] = grp["가격"].mean()
        grid[d, zi, ti, ci, 3] = grp["에이전트ID"].nunique()

    return _normalize_cf(grid), days_present


# ── 메인 inference 함수 ──────────────────────────────────────────────────────

def run_inference(
    df: pd.DataFrame,
    zone_stats: Dict[str, Any],
    hourly_stats: List[Dict[str, Any]],
    operating_hours: int,
) -> Dict:
    """
    Transformer + Crossformer 를 실행해 5개 KPI를 반환한다.

    zone_stats:   { zoneId: { visitors, buyers, label } }  (simul 실측 누적값)
    hourly_stats: [{ hour (0-base), visitors, buyers }]     (simul 실측, 시간대별 누적)
    """
    if _S.tf_model is None or _S.cf_model is None:
        raise RuntimeError("모델이 로드되지 않았습니다. load_models()를 먼저 호출하세요.")

    TF_WIN = _S.tf_ckpt.get("window_size", 7)
    CF_WIN = _S.cf_ckpt["cfg_data"].get("window_size", 5)

    # ── Transformer grid ──────────────────────────────────────────────────
    tf_grid, tf_days, unknown_zones = _build_tf_grid(df)
    D_tf = len(tf_days)

    if tf_grid is None or D_tf < TF_WIN:
        raise ValueError(
            f"데이터 부족: Transformer는 {TF_WIN}일 이상 필요합니다 "
            f"(현재 {D_tf}일). 시뮬레이션을 더 길게 실행해 주세요."
        )

    Z = _S.tf_ckpt["n_zones"]
    T = _S.tf_ckpt["n_hours"]
    F = _S.tf_ckpt["n_features"]

    x_tf  = torch.from_numpy(tf_grid[-TF_WIN:].reshape(1, TF_WIN, Z * T, F))
    dow_tf = torch.tensor([[(d - 1) % 7 for d in tf_days[-TF_WIN:]]], dtype=torch.long)

    with torch.no_grad():
        pred_tf_log = _S.tf_model(x_tf.to(_S.device), dow_tf.to(_S.device))  # [1, 1, Z*T]
    pred_tf = np.expm1(pred_tf_log.cpu().numpy().reshape(Z, T)).clip(0)        # [Z, T]

    # ── Crossformer grid ──────────────────────────────────────────────────
    cf_grid, cf_days = _build_cf_grid(df)
    D_cf = len(cf_days)

    if cf_grid is None or D_cf < CF_WIN:
        raise ValueError(
            f"데이터 부족: Crossformer는 {CF_WIN}일 이상 필요합니다 "
            f"(현재 {D_cf}일). 시뮬레이션을 더 길게 실행해 주세요."
        )

    C = _S.cf_ckpt["n_classes"]
    x_cf  = torch.from_numpy(cf_grid[-CF_WIN:].reshape(1, CF_WIN, Z * T * C, 4))
    dow_cf = torch.tensor([[(d - 1) % 7 for d in cf_days[-CF_WIN:]]], dtype=torch.long)

    with torch.no_grad():
        pred_cf_buy_log, pred_cf_dwell = _S.cf_model(
            x_cf.to(_S.device), dow_cf.to(_S.device)
        )  # [1,1,Z*T*C], [1,1,Z*T]

    pred_cf_dwell = pred_cf_dwell.cpu().numpy().reshape(Z, T).clip(0)  # [Z, T]

    # ── KPI 산출 ──────────────────────────────────────────────────────────
    total_days    = max(tf_days) - min(tf_days) + 1
    prediction_day = max(tf_days) + 1

    zone2idx    = _S.tf_ckpt["zone2idx"]
    idx2zone    = {v: k for k, v in zone2idx.items()}
    hour2idx    = _S.tf_ckpt["hour2idx"]
    idx2hour    = {v: k for k, v in hour2idx.items()}
    start_hour  = max(6, 22 - operating_hours)

    # 시간대별 방문자/구매자 (0-base simul 인덱스 → 전체 기간 누적)
    hour_visitors: Dict[int, int] = {}
    hour_buyers:   Dict[int, int] = {}
    for hs in hourly_stats:
        h = hs.get("hour", 0)
        hour_visitors[h] = hour_visitors.get(h, 0) + hs.get("visitors", 0)
        hour_buyers[h]   = hour_buyers.get(h,   0) + hs.get("buyers",   0)

    # ── 전체 스케일 보정 ──────────────────────────────────────────────────
    # 학습 데이터 스케일과 simul 스케일이 다를 때 convRate=100%가 되는 문제 방지.
    # 모델 예측의 절대 총합을 실제 일평균 구매수에 맞게 rescale한다.
    # → 모델은 구역/시간대 간 상대적 분포만 제공하고, 총합은 실측에 고정.
    total_actual_buyers = sum(zs.get("buyers", 0) for zs in zone_stats.values())
    total_actual_daily  = total_actual_buyers / max(total_days, 1)
    total_pred_raw      = float(pred_tf.sum())
    if total_pred_raw > 1e-6 and total_actual_daily > 0:
        pred_tf = pred_tf * (total_actual_daily / total_pred_raw)

    # KPI #1, #3, #7, #8 — 구역별
    zone_buy_pred   = pred_tf.sum(axis=1)        # [Z] 예측 일별 구매수 합
    zone_dwell_pred = pred_cf_dwell.mean(axis=1) # [Z] 예측 평균 체류시간

    max_dwell = float(zone_dwell_pred.max()) or 1.0
    max_conv  = 1e-6  # 먼저 수집 후 재계산

    zone_kpi_raw = []
    for zi in range(Z):
        zone_id = idx2zone.get(zi)
        if zone_id is None:
            continue
        zs = zone_stats.get(zone_id, {})
        label    = zs.get("label", zone_id)
        visitors = zs.get("visitors", 0)

        avg_pred_buy = float(zone_buy_pred[zi])
        avg_vis      = visitors / max(total_days, 1)
        conv_rate    = min(avg_pred_buy / avg_vis, 1.0) if avg_vis > 0 else 0.0
        avg_dwell    = max(float(zone_dwell_pred[zi]), 0.0)

        max_conv = max(max_conv, conv_rate)

        zone_kpi_raw.append({
            "zoneId":   zone_id,
            "label":    label,
            "convRate": conv_rate,
            "avgDwell": avg_dwell,
        })

    zone_kpi = []
    for z in zone_kpi_raw:
        eff   = z["convRate"] / z["avgDwell"] if z["avgDwell"] > 0 else 0.0
        ineff = (z["avgDwell"] / max_dwell) * (1 - z["convRate"] / max_conv) if max_conv > 1e-6 else 0.0
        zone_kpi.append({
            "zoneId":      z["zoneId"],
            "label":       z["label"],
            "convRate":    round(z["convRate"], 4),
            "avgDwell":    round(z["avgDwell"], 2),
            "efficiency":  round(eff, 6),
            "inefficiency": round(ineff, 4),
        })

    # KPI #6 — 시간대별
    # pred_tf[:, ti].sum() = zone-purchase 이벤트 합산 (방문자 1명이 3구역 구매 → 3건)
    # 이를 고유 방문자 수로 나누면 비율 > 1 → 100% 고정 문제 발생.
    # 해결: 실측 hourly convRate(구매자/방문자)를 기준으로, 모델 예측의 시간대별
    # 상대 비중으로 스케일링 — 총합 고정 후 분포만 모델이 제공.
    total_zone_pred = float(pred_tf.sum())
    hour_kpi = []
    for ti in range(T):
        actual_hour = idx2hour.get(ti)
        if actual_hour is None:
            continue
        simul_h     = actual_hour - start_hour
        vis_total   = hour_visitors.get(simul_h, 0)
        buy_total   = hour_buyers.get(simul_h, 0)

        # 실측 시간대별 전환율 (분자·분모 모두 고유 인원 → 항상 [0,1])
        actual_conv = buy_total / max(vis_total, 1) if vis_total > 0 else 0.0

        # 모델이 예측한 이 시간대의 zone-구매 비중 (uniform=1.0, 바쁜 시간대>1.0)
        # floor=0.4: 마감 시간대에 모델이 과도하게 0으로 수렴하는 것 방지
        pred_hour_zone = float(pred_tf[:, ti].sum())
        if total_zone_pred > 1e-6:
            relative_scale = max(0.4, (pred_hour_zone / total_zone_pred) * T)
        else:
            relative_scale = 1.0

        conv_rate = min(actual_conv * relative_scale, 1.0)
        hour_kpi.append({
            "hour":     ti,
            "label":    f"{actual_hour}시",
            "convRate": round(conv_rate, 4),
        })

    warnings = []
    if total_pred_raw < 1e-6:
        warnings.append("모델 예측값이 0에 가깝습니다. 입력 데이터를 확인해 주세요.")

    return {
        "predictionDay": int(prediction_day),
        "windowDays":    TF_WIN,
        "zoneKPI":       zone_kpi,
        "hourKPI":       hour_kpi,
        "models": {
            "transformer": {"reportedZoneMAPE": 4.0, "reportedHourMAPE": 5.4},
            "crossformer": {"reportedDwellMAPE": 8.6, "reportedEffMAPE": 3.9},
        },
        "unknownZones": unknown_zones,
        "warnings":     warnings,
    }
