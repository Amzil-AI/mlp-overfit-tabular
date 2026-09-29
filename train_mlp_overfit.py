"""Séance 1 atelier — deliberately overfit a small MLP (non-finance tabular).

1) California Housing regression (MSE)
2) Breast Cancer binary classification (BCE)

Teaching goal: train loss drops while validation gap grows. No dropout / weight decay.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"

SEED = 42
BATCH_SIZE = 64
EPOCHS = 120
LR = 1e-3
HIDDEN = (256, 256, 128)  # intentionally wide vs sample size
VAL_SIZE = 0.25


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loaders(
    X: np.ndarray, y: np.ndarray, *, classification: bool
) -> tuple[DataLoader, DataLoader, int]:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SIZE, random_state=SEED, stratify=y if classification else None
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    y_train = y_train.astype(np.float32).reshape(-1, 1)
    y_val = y_val.astype(np.float32).reshape(-1, 1)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, X_train.shape[1]


@torch.no_grad()
def epoch_loss(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, classification: bool
) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        logits = model(xb)
        if classification:
            loss = criterion(logits, yb)
        else:
            loss = criterion(logits, yb)
        total += float(loss.item()) * len(xb)
        n += len(xb)
    return total / max(n, 1)


def train_task(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
) -> dict:
    train_loader, val_loader, n_in = make_loaders(X, y, classification=classification)
    model = MLP(n_in, n_out=1)
    criterion = nn.BCEWithLogitsLoss() if classification else nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)  # no weight decay

    train_hist: list[float] = []
    val_hist: list[float] = []

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        train_hist.append(epoch_loss(model, train_loader, criterion, classification))
        val_hist.append(epoch_loss(model, val_loader, criterion, classification))

    gap = [v - t for t, v in zip(train_hist, val_hist)]
    # first epoch where validation stays above train by a clear margin and keeps growing
    diverge_epoch = None
    for i in range(5, len(gap)):
        if gap[i] > max(0.02 * (val_hist[0] + 1e-8), 1e-4) and gap[i] > gap[i - 5]:
            diverge_epoch = i + 1
            break

    return {
        "name": name,
        "task": "classification" if classification else "regression",
        "loss": "BCEWithLogits" if classification else "MSE",
        "n_train": int(len(train_loader.dataset)),
        "n_val": int(len(val_loader.dataset)),
        "epochs": EPOCHS,
        "train_loss_final": train_hist[-1],
        "val_loss_final": val_hist[-1],
        "gap_final": gap[-1],
        "diverge_epoch_approx": diverge_epoch,
        "train_hist": train_hist,
        "val_hist": val_hist,
    }


def plot_curves(result: dict, path: Path) -> None:
    epochs = np.arange(1, len(result["train_hist"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, result["train_hist"], label="Train", color="#1f4e79", lw=2)
    ax.plot(epochs, result["val_hist"], label="Validation", color="#c0392b", lw=2)
    if result["diverge_epoch_approx"]:
        ax.axvline(
            result["diverge_epoch_approx"],
            color="#7f8c8d",
            ls="--",
            lw=1.2,
            label=f"Écart visible ≈ époque {result['diverge_epoch_approx']}",
        )
    ax.set_xlabel("Époque")
    ax.set_ylabel(result["loss"])
    ax.set_title(f"{result['name']} — train vs validation")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    set_seed()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    housing = fetch_california_housing()
    # subsample so a wide MLP can memorize train faster (clearer teaching plot)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(housing.data), size=2500, replace=False)
    reg = train_task(
        "California Housing (régression)",
        housing.data[idx],
        housing.target[idx],
        classification=False,
    )
    plot_curves(reg, FIG_DIR / "regression_curves.png")

    cancer = load_breast_cancer()
    clf = train_task(
        "Breast Cancer (classification binaire)",
        cancer.data,
        cancer.target.astype(np.float32),
        classification=True,
    )
    plot_curves(clf, FIG_DIR / "classification_curves.png")

    summary = {
        "architecture": {"hidden": list(HIDDEN), "activation": "ReLU", "regularization": "none"},
        "regression": {k: v for k, v in reg.items() if k not in ("train_hist", "val_hist")},
        "classification": {k: v for k, v in clf.items() if k not in ("train_hist", "val_hist")},
        "histories": {
            "regression": {"train": reg["train_hist"], "val": reg["val_hist"]},
            "classification": {"train": clf["train_hist"], "val": clf["val_hist"]},
        },
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("Regression  final train/val:", reg["train_loss_final"], reg["val_loss_final"])
    print("Classification final train/val:", clf["train_loss_final"], clf["val_loss_final"])
    print("Wrote", FIG_DIR / "regression_curves.png")
    print("Wrote", FIG_DIR / "classification_curves.png")
    print("Wrote", OUT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
