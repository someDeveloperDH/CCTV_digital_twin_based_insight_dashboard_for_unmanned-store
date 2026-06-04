import math
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import PurchaseGridDataset
from model import PurchaseGridTransformer


def load_config(path: str = "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = load_config()
    dc, mc, tc = cfg["data"], cfg["model"], cfg["train"]
    set_seed(42)

    device_str = tc.get("device", "auto")
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if device_str == "auto" else torch.device(device_str))

    W, H = dc["window_size"], dc["horizon"]
    train_end = dc["train_days"]
    val_end = train_end + dc["val_days"]

    # train으로 정규화 통계 fit → val에 그대로 적용
    train_ds = PurchaseGridDataset(dc["purchases_csv"], W, H, day_range=(1, train_end))
    val_ds = PurchaseGridDataset(dc["purchases_csv"], W, H,
                                  day_range=(train_end + 1, val_end), stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"])

    sample_x, _, _ = train_ds[0]
    n_features = sample_x.shape[2]

    model = PurchaseGridTransformer(
        n_zones=train_ds.n_zones,
        n_hours=train_ds.n_hours,
        n_features=n_features,
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        dropout=mc["dropout"],
        horizon=H,
        max_window=max(W, 64),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tc["lr"], weight_decay=tc["weight_decay"])
    criterion = nn.SmoothL1Loss(beta=0.5)

    epochs = tc["epochs"]
    warmup = tc["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(tc["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)

    print(f"Device: {device}  |  zones={train_ds.n_zones}  hours={train_ds.n_hours}  "
          f"tokens={train_ds.n_tokens}  features={n_features}")
    print(f"Train samples={len(train_ds)}  Val samples={len(val_ds)}\n")

    best_val = float("inf")
    no_improve = 0
    patience = tc["patience"]
    grad_clip = tc.get("grad_clip", 1.0)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y, dow in train_loader:
            x, y, dow = x.to(device), y.to(device), dow.to(device)
            pred = model(x, dow)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, dow in val_loader:
                x, y, dow = x.to(device), y.to(device), dow.to(device)
                val_loss += criterion(model(x, dow), y).item() * x.size(0)
        val_loss /= max(len(val_ds), 1)

        scheduler.step()
        improved = val_loss < best_val - 1e-5

        if epoch % 5 == 0 or improved:
            marker = " *" if improved else ""
            print(f"Epoch {epoch:3d} | lr={scheduler.get_last_lr()[0]:.5f} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f}{marker}")

        if improved:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_loss": float(val_loss),
                "zone2idx": train_ds.zone2idx,
                "hour2idx": train_ds.hour2idx,
                "cls2idx": train_ds.cls2idx,
                "stats": train_ds.stats,
                "n_zones": train_ds.n_zones,
                "n_hours": train_ds.n_hours,
                "n_features": n_features,
                "cfg_model": mc,
                "cfg_data": dc,
            }, ckpt_dir / "best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\nBest val loss: {best_val:.4f}  →  {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
