# -*- coding: utf-8 -*-
"""
train.py
========
最终稳定版训练与评估。

LSTM / Transformer：
    纯 MSE，作为干净基线。

Stable-DR-CNN-Transformer：
    以 MSE 为主；
    附加很小权重的峰谷加权 Huber、差分一致性、趋势一致性损失。
这样既保留创新点，又尽量贴近最终评价指标 MSE/MAE。
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import build_model


class StableInnovationLoss(nn.Module):
    """
    L = MSE
        + peak_beta * PeakWeightedHuber
        + diff_gamma * Huber(diff(pred), diff(target))
        + trend_lambda * Huber(MA(pred), MA(target))

    其中：
    - MSE 是主项，保证和最终评价指标一致；
    - peak/diff/trend 都是小权重正则；
    - trend consistency 比直接追每日差分更稳定，尤其适合 365 天。
    """
    def __init__(
        self,
        peak_beta: float,
        diff_gamma: float,
        trend_lambda: float,
        trend_window: int,
        max_weight: float = 1.8,
        huber_beta: float = 0.6,
    ):
        super().__init__()
        self.peak_beta = float(peak_beta)
        self.diff_gamma = float(diff_gamma)
        self.trend_lambda = float(trend_lambda)
        self.trend_window = int(trend_window)
        self.max_weight = float(max_weight)
        self.huber_beta = float(huber_beta)
        self.mse = nn.MSELoss()

    @staticmethod
    def _moving_average(x, window: int):
        # x: (B, H)
        if window <= 1 or x.size(1) < window:
            return x
        kernel = torch.ones(1, 1, window, device=x.device, dtype=x.dtype) / float(window)
        return F.conv1d(x.unsqueeze(1), kernel, stride=1).squeeze(1)

    def forward(self, pred, target):
        base_mse = self.mse(pred, target)

        # 峰谷项：权重很小，只是提醒模型关注极端值，不主导训练。
        if self.peak_beta > 0:
            point_huber = F.smooth_l1_loss(
                pred,
                target,
                reduction="none",
                beta=self.huber_beta,
            )
            weight = 1.0 + torch.abs(target.detach())
            weight = torch.clamp(weight, min=1.0, max=self.max_weight)
            peak_loss = torch.mean(weight * point_huber)
        else:
            peak_loss = torch.tensor(0.0, device=pred.device)

        # 差分项：小权重，避免完全平滑。
        if self.diff_gamma > 0 and pred.size(1) > 1:
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            diff_loss = F.smooth_l1_loss(
                pred_diff,
                target_diff,
                reduction="mean",
                beta=self.huber_beta,
            )
        else:
            diff_loss = torch.tensor(0.0, device=pred.device)

        # 趋势一致性：约束滑动平均趋势，替代激进追峰。
        if self.trend_lambda > 0 and pred.size(1) >= self.trend_window:
            pred_ma = self._moving_average(pred, self.trend_window)
            target_ma = self._moving_average(target, self.trend_window)
            trend_loss = F.smooth_l1_loss(
                pred_ma,
                target_ma,
                reduction="mean",
                beta=self.huber_beta,
            )
        else:
            trend_loss = torch.tensor(0.0, device=pred.device)

        return (
            base_mse
            + self.peak_beta * peak_loss
            + self.diff_gamma * diff_loss
            + self.trend_lambda * trend_loss
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_innovation_model(model_name: str) -> bool:
    return model_name.lower() in (
        "stable-dr-cnn-transformer",
        "stable-stable-dr-cnn-transformer",
        "dr-cnn-transformer",
        "drcnntransformer",
        "dr_cnn_transformer",
        "cnn-transformer",
    )


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x).detach().cpu().numpy()
        preds.append(out)
        trues.append(y.numpy())

    if not preds:
        return np.empty((0,)), np.empty((0,))

    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def _inverse(arr, target_scaler):
    flat = arr.reshape(-1, 1)
    inv = target_scaler.inverse_transform(flat).reshape(arr.shape)
    return inv


def evaluate(model, loader, target_scaler, device):
    preds_s, trues_s = _predict(model, loader, device)
    if preds_s.size == 0:
        return float("nan"), float("nan"), None, None

    preds = _inverse(preds_s, target_scaler)
    trues = _inverse(trues_s, target_scaler)

    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    return mse, mae, preds, trues


def train_one_run(model_name, loaders, target_scaler, feature_dim, horizon, cfg, seed):
    set_seed(seed)
    device = get_device(cfg)
    train_loader, val_loader, test_loader = loaders[:3]

    model = build_model(model_name, feature_dim, horizon, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if is_innovation_model(model_name):
        criterion = StableInnovationLoss(
            peak_beta=cfg.get_peak_weight_beta(horizon),
            diff_gamma=cfg.get_diff_loss_gamma(horizon),
            trend_lambda=cfg.get_trend_loss_lambda(horizon),
            trend_window=cfg.get_trend_ma_window(horizon),
            max_weight=cfg.peak_weight_max,
            huber_beta=cfg.huber_beta,
        )
        loss_name = (
            "MSE+StableReg("
            f"peak={cfg.get_peak_weight_beta(horizon):g}, "
            f"diff={cfg.get_diff_loss_gamma(horizon):g}, "
            f"trend={cfg.get_trend_loss_lambda(horizon):g})"
        )
    else:
        criterion = nn.MSELoss()
        loss_name = "MSE"

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= max(1, len(train_loader.dataset))

        vp, vt = _predict(model, val_loader, device)
        if vp.size:
            with torch.no_grad():
                val_loss = criterion(
                    torch.from_numpy(vp).to(device),
                    torch.from_numpy(vt).to(device),
                ).item()
        else:
            val_loss = epoch_loss

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % max(1, cfg.epochs // 5) == 0 or epoch == 1:
            print(
                f"      [{model_name:24s} H={horizon:3d} seed={seed}] "
                f"epoch {epoch:3d}/{cfg.epochs} train_loss={epoch_loss:.4f} "
                f"val_loss={val_loss:.4f} loss={loss_name}"
            )

        if patience >= cfg.early_stop_patience:
            print(f"      早停于 epoch {epoch}（val 连续 {patience} 次未提升）")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    mse, mae, preds, trues = evaluate(model, test_loader, target_scaler, device)
    return {"mse": mse, "mae": mae, "preds": preds, "trues": trues}
