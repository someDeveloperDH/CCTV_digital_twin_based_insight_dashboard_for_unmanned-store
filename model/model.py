import torch
import torch.nn as nn


class SpaceTimeBlock(nn.Module):
    """
    Spatial attention (한 시점 내 Z*T 토큰끼리) +
    Temporal attention (같은 토큰의 L 시점끼리) +
    FFN
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_s = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_f = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N, d]
        B, L, N, d = x.shape

        xs = x.reshape(B * L, N, d)
        xs2, _ = self.spatial_attn(xs, xs, xs)
        x = self.norm_s(x + self.drop(xs2.reshape(B, L, N, d)))

        xt = x.permute(0, 2, 1, 3).reshape(B * N, L, d)
        xt2, _ = self.temporal_attn(xt, xt, xt)
        x = self.norm_t(x + self.drop(xt2.reshape(B, N, L, d).permute(0, 2, 1, 3)))

        x = self.norm_f(x + self.drop(self.ffn(x)))
        return x


class PurchaseGridTransformer(nn.Module):
    """
    구역×시간대 grid 시계열 → 다음 horizon일의 셀별 log1p(구매수) 예측

    입력:
      x:   [B, L, N, F]    (정규화된 피처)
      dow: [B, L]          (요일 0~6)
    출력:
      [B, horizon, N]      (log1p 공간의 예측값)

    임베딩 분리: zone(Z) + hour(T) + dow(7) + 일차 위치(L)
    """

    def __init__(self, n_zones: int, n_hours: int, n_features: int,
                 d_model: int = 96, n_heads: int = 4, n_layers: int = 3,
                 dropout: float = 0.1, horizon: int = 1, max_window: int = 64):
        super().__init__()
        self.n_zones = n_zones
        self.n_hours = n_hours

        self.input_proj = nn.Linear(n_features, d_model)
        self.zone_emb = nn.Embedding(n_zones, d_model)
        self.hour_emb = nn.Embedding(n_hours, d_model)
        self.dow_emb = nn.Embedding(7, d_model)
        self.pos_emb = nn.Embedding(max_window, d_model)

        self.blocks = nn.ModuleList([
            SpaceTimeBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N, F], dow: [B, L]
        B, L, N, _ = x.shape
        Z, T = self.n_zones, self.n_hours

        h = self.input_proj(x)                                            # [B, L, N, d]

        # zone × hour 분리 임베딩
        zone_ids = torch.arange(Z, device=x.device).repeat_interleave(T)   # [N]
        hour_ids = torch.arange(T, device=x.device).repeat(Z)              # [N]
        token_emb = self.zone_emb(zone_ids) + self.hour_emb(hour_ids)      # [N, d]
        h = h + token_emb.view(1, 1, N, -1)

        # 일차 위치
        pos_ids = torch.arange(L, device=x.device).view(1, L, 1)           # [1, L, 1]
        h = h + self.pos_emb(pos_ids)

        # 요일 임베딩
        dow_e = self.dow_emb(dow).unsqueeze(2)                              # [B, L, 1, d]
        h = h + dow_e

        for block in self.blocks:
            h = block(h)

        out = self.head(h[:, -1, :, :])      # [B, N, horizon]
        return out.permute(0, 2, 1)            # [B, horizon, N]
