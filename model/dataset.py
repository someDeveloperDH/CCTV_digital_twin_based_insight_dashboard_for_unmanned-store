import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PurchaseGridDataset(Dataset):
    """
    batch_purchases CSV → (일차, 구역, 시간대) 집계 grid → 슬라이딩 윈도우

    입력 텐서: [window_size, Z*T, F]  (정규화된 피처)
    타겟 텐서: [horizon, Z*T]          (log1p(구매 이벤트 수))

    카테고리 매핑은 항상 전체 CSV 기준 → train/val/test 인덱스 일관성 보장.
    정규화 통계(mean/std)는 train 샘플에서 한 번 fit 후 stats 인자로 공유.
    """

    N_FEATURES = 8  # 구매수, 체류시간, 가격, 고객수, 고객군 비중 4채널

    def __init__(self, purchases_path: str, window_size: int = 7, horizon: int = 1,
                 day_range: tuple = None, stats: dict = None):
        df_all = pd.read_csv(purchases_path)
        df_all["hour"] = df_all["시간대"].str.replace("시", "").astype(int)

        # 카테고리 매핑 (전체 CSV 기준)
        self.zones = sorted(df_all["구역ID"].unique())
        self.hours = sorted(df_all["hour"].unique())
        self.classes = sorted(df_all["고객분류"].unique())
        self.zone2idx = {z: i for i, z in enumerate(self.zones)}
        self.hour2idx = {h: i for i, h in enumerate(self.hours)}
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}
        self.n_zones = len(self.zones)
        self.n_hours = len(self.hours)
        self.n_tokens = self.n_zones * self.n_hours

        # day_range 필터
        df = df_all if day_range is None else df_all[df_all["일차"].between(day_range[0], day_range[1])]
        days_present = sorted(df["일차"].unique())
        n_days = len(days_present)
        self.day2seq = {d: i for i, d in enumerate(days_present)}
        self.seq2day = {i: d for d, i in self.day2seq.items()}

        # 원본 grid 빌드 (정규화 전)
        self.grid_raw = self._build_grid(df, n_days)  # [D, Z, T, F]

        # 정규화 통계 fit/transform
        if stats is None:
            self.stats = self._fit_stats(self.grid_raw)
        else:
            self.stats = stats

        self.grid = self._normalize(self.grid_raw, self.stats)        # 입력용 (정규화)
        self.target_raw = self.grid_raw[..., 0]                       # 구매 이벤트 수
        self.target = np.log1p(self.target_raw).astype(np.float32)    # log1p 타겟

        self.window_size = window_size
        self.horizon = horizon
        max_start = n_days - window_size - horizon + 1
        self.indices = list(range(max(0, max_start)))

    def _build_grid(self, df: pd.DataFrame, n_days: int) -> np.ndarray:
        Z, T = self.n_zones, self.n_hours
        F = self.N_FEATURES
        grid = np.zeros((n_days, Z, T, F), dtype=np.float32)

        for (day, zone, hour), grp in df.groupby(["일차", "구역ID", "hour"]):
            if day not in self.day2seq:
                continue
            d = self.day2seq[day]
            zi = self.zone2idx[zone]
            ti = self.hour2idx[hour]

            grid[d, zi, ti, 0] = len(grp)
            grid[d, zi, ti, 1] = grp["체류시간"].mean()
            grid[d, zi, ti, 2] = grp["가격"].mean()
            grid[d, zi, ti, 3] = grp["에이전트ID"].nunique()
            total = len(grp)
            for cls, grp_cls in grp.groupby("고객분류"):
                ci = self.cls2idx.get(cls)
                if ci is not None:
                    grid[d, zi, ti, 4 + ci] = len(grp_cls) / total

        return grid

    @staticmethod
    def _fit_stats(grid: np.ndarray) -> dict:
        """count류 (0,3): log1p 후 mean/std. 연속값 (1,2): raw mean/std. 비중 (4~7): 그대로."""
        F = grid.shape[-1]
        flat = grid.reshape(-1, F)
        # nonzero 셀에서만 통계 (0이 많아서 평균이 왜곡되는 것 방지)
        nz_mask = flat[:, 0] > 0  # 활성 셀
        active = flat[nz_mask]

        means = np.zeros(F, dtype=np.float32)
        stds = np.ones(F, dtype=np.float32)
        log_indices = {0, 3}  # count 피처
        skip_indices = {4, 5, 6, 7}  # 비중은 이미 [0,1]

        for i in range(F):
            if i in skip_indices:
                continue
            col = active[:, i]
            if i in log_indices:
                col = np.log1p(col)
            means[i] = col.mean() if len(col) else 0.0
            stds[i] = col.std() + 1e-6 if len(col) else 1.0

        return {"means": means, "stds": stds, "log_indices": list(log_indices),
                "skip_indices": list(skip_indices)}

    @staticmethod
    def _normalize(grid: np.ndarray, stats: dict) -> np.ndarray:
        out = grid.copy()
        means, stds = stats["means"], stats["stds"]
        log_idx = set(stats["log_indices"])
        skip_idx = set(stats["skip_indices"])
        F = grid.shape[-1]
        for i in range(F):
            if i in skip_idx:
                continue
            if i in log_idx:
                out[..., i] = np.log1p(out[..., i])
            out[..., i] = (out[..., i] - means[i]) / stds[i]
        return out.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        x = self.grid[start: start + self.window_size]                            # [L, Z, T, F]
        y = self.target[start + self.window_size:
                        start + self.window_size + self.horizon]                  # [H, Z, T]

        # day-of-week (0~6) — 일차를 7로 나눈 나머지
        days = [self.seq2day[start + i] for i in range(self.window_size)]
        dow = np.array([(d - 1) % 7 for d in days], dtype=np.int64)               # [L]

        L, Z, T, F = x.shape
        x = x.reshape(L, Z * T, F)
        y = y.reshape(self.horizon, Z * T)

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(dow)
