import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

from dataset import PurchaseGridDataset
from model import PurchaseGridTransformer


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    dc, tc = cfg["data"], cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(tc["checkpoint_dir"]) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    W, H = dc["window_size"], dc["horizon"]
    train_end = dc["train_days"]
    val_days = dc["val_days"]
    test_start = train_end + val_days + 1

    test_ds = PurchaseGridDataset(dc["purchases_csv"], W, H,
                                   day_range=(test_start, 9999), stats=ckpt["stats"])
    test_loader = DataLoader(test_ds, batch_size=64)

    mc = ckpt["cfg_model"]
    model = PurchaseGridTransformer(
        n_zones=ckpt["n_zones"],
        n_hours=ckpt["n_hours"],
        n_features=ckpt["n_features"],
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        dropout=mc["dropout"],
        horizon=H,
        max_window=max(W, 64),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds_log, trues_log = [], []
    with torch.no_grad():
        for x, y, dow in test_loader:
            preds_log.append(model(x.to(device), dow.to(device)).cpu().numpy())
            trues_log.append(y.numpy())
    preds_log = np.concatenate(preds_log)   # log1p 공간
    trues_log = np.concatenate(trues_log)

    # log1p 역변환 → 원래 count 공간
    preds = np.expm1(preds_log).clip(min=0)
    trues = np.expm1(trues_log)

    mae = np.abs(preds - trues).mean()

    print(f"[셀 단위] MAE={mae:.4f}  (test samples={len(preds)})")

    Z = ckpt["n_zones"]
    T = ckpt["n_hours"]
    p = preds.reshape(-1, H, Z, T)
    t = trues.reshape(-1, H, Z, T)

    zone_pred = p.sum(axis=3).mean(axis=(0, 1))
    zone_true = t.sum(axis=3).mean(axis=(0, 1))
    hour_pred = p.sum(axis=2).mean(axis=(0, 1))
    hour_true = t.sum(axis=2).mean(axis=(0, 1))

    zone_mae = np.abs(zone_pred - zone_true).mean()
    hour_mae = np.abs(hour_pred - hour_true).mean()
    zone_mape = (np.abs(zone_pred - zone_true) / (zone_true + 1e-6)).mean() * 100
    hour_mape = (np.abs(hour_pred - hour_true) / (hour_true + 1e-6)).mean() * 100
    z_rho = spearmanr(zone_pred, zone_true).statistic
    t_rho = spearmanr(hour_pred, hour_true).statistic

    print(f"\n[구역별]   MAE={zone_mae:.2f}  MAPE={zone_mape:.1f}%  순위 ρ={z_rho:.3f}")
    print(f"[시간대별] MAE={hour_mae:.2f}  MAPE={hour_mape:.1f}%  순위 ρ={t_rho:.3f}")

    # Sanity: zone_kpi와 비교
    zone_kpi = pd.read_csv(dc["zone_kpi_csv"])
    idx2zone = {v: k for k, v in ckpt["zone2idx"].items()}
    df_raw = pd.read_csv(dc["purchases_csv"])
    id2name = df_raw.drop_duplicates("구역ID").set_index("구역ID")["구역명"].to_dict()

    pred_pct = zone_pred / zone_pred.sum() * 100
    true_pct = zone_true / zone_true.sum() * 100

    print("\n[sanity] 구역별 구매 비중  pred vs 실제(test) vs kpi집계(4달)")
    print(f"  {'구역ID':10s}  {'구역명':8s}  pred(%)  true(%)  kpi(%)")
    for zi in range(Z):
        zone_id = idx2zone.get(zi, str(zi))
        zone_name = id2name.get(zone_id, zone_id)
        row = zone_kpi[zone_kpi["구역"] == zone_name]
        ref = row["구매고객비중(%)"].values[0] if len(row) else float("nan")
        print(f"  {zone_id:10s}  {zone_name:8s}  {pred_pct[zi]:6.1f}   {true_pct[zi]:6.1f}   {ref:6.1f}")

    # 시간대별 pred vs true 표
    idx2hour = {v: k for k, v in ckpt["hour2idx"].items()}
    print("\n[시간대별]  pred vs 실제(test)")
    print(f"  {'시간':6s}  pred    true")
    for ti in range(T):
        h = idx2hour.get(ti, ti)
        print(f"  {h:>2}시   {hour_pred[ti]:6.1f}  {hour_true[ti]:6.1f}")

    # baseline 비교: window 마지막 일을 그대로 예측
    print("\n[baseline] persistence (마지막 일 = 다음 일):")
    p_last = []
    t_y = []
    grid = test_ds.target_raw  # [D, Z, T]  log1p 안 한 원본
    for s in test_ds.indices:
        p_last.append(grid[s + W - 1])
        t_y.append(grid[s + W])
    p_last = np.stack(p_last)
    t_y = np.stack(t_y)
    base_mae = np.abs(p_last - t_y).mean()
    base_zone_pred = p_last.sum(axis=2).mean(axis=0)
    base_zone_true = t_y.sum(axis=2).mean(axis=0)
    base_z_rho = spearmanr(base_zone_pred, base_zone_true).statistic
    print(f"  cell MAE={base_mae:.4f}  zone ρ={base_z_rho:.3f}")
    print(f"  → 모델이 baseline 대비 cell MAE {(base_mae - mae) / base_mae * 100:+.1f}% 개선")


if __name__ == "__main__":
    main()
