# -*- coding: utf-8 -*-
"""
models.py
=========
最终稳定版模型文件。

包含：
1. LSTM
2. Transformer
3. Stable-DR-CNN-Transformer

Stable-DR-CNN-Transformer 的设计：
- 趋势分支：Moving Average 分解后的 trend 输入 Transformer；
- 残差分支：residual 输入轻量 CNN，不再使用第二个 Transformer，降低过拟合；
- 残差门控：根据 trend/residual 表征自动调节残差注入强度；
- 显式输出：prediction = trend_prediction + bounded_residual_prediction；
- 残差用 tanh 限幅，90天残差稍强，365天残差较弱。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PointwiseHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, future_feats):
        return self.net(future_feats).squeeze(-1)


class LSTMForecast(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, horizon, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = PointwiseHead(hidden_dim, dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        future_out = out[:, -self.horizon:, :]
        return self.head(future_out)


class TransformerForecast(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, n_layers, horizon, max_len=1024, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = PointwiseHead(d_model, dropout)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        future_h = h[:, -self.horizon:, :]
        return self.head(future_h)


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 15):
        super().__init__()
        if kernel_size < 1:
            kernel_size = 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x):
        if self.kernel_size == 1:
            return x, torch.zeros_like(x)

        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (self.pad, self.pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        trend = trend.transpose(1, 2)
        residual = x - trend
        return trend, residual


class StableDRCNNTransformerForecast(nn.Module):
    """
    Stable Decomposition-Residual CNN-Transformer.

    与上一版相比：
    - 残差分支去掉 Transformer，只保留 CNN，减少参数和过拟合；
    - trend branch 是主干；
    - residual branch 只做小幅修正；
    - bounded_residual = residual_scale * tanh(residual_raw)。
    """
    def __init__(
        self,
        input_dim,
        d_model,
        n_heads,
        n_layers,
        horizon,
        kernel_size=3,
        decomp_kernel=15,
        residual_output_scale=0.5,
        max_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.residual_output_scale = float(residual_output_scale)

        self.decomp = MovingAverageDecomposition(decomp_kernel)

        # Trend branch
        self.trend_proj = nn.Linear(input_dim, d_model)
        self.trend_pos = PositionalEncoding(d_model, max_len)
        trend_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.trend_encoder = nn.TransformerEncoder(trend_layer, num_layers=n_layers)

        # Lightweight residual branch: CNN only
        self.resid_proj = nn.Linear(input_dim, d_model)
        conv_pad = kernel_size // 2
        self.resid_cnn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=conv_pad),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size, padding=conv_pad),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Residual gate
        self.residual_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.trend_head = PointwiseHead(d_model, dropout)
        self.residual_head = PointwiseHead(d_model, dropout)

    def forward(self, x):
        trend_x, resid_x = self.decomp(x)

        trend_h = self.trend_proj(trend_x)
        trend_h = self.trend_pos(trend_h)
        trend_h = self.trend_encoder(trend_h)

        resid_h = self.resid_proj(resid_x)
        resid_h = resid_h.transpose(1, 2)
        resid_h = self.resid_cnn(resid_h)
        resid_h = resid_h.transpose(1, 2)

        gate = self.residual_gate(torch.cat([trend_h, resid_h], dim=-1))
        gated_resid = gate * resid_h

        trend_future = trend_h[:, -self.horizon:, :]
        resid_future = gated_resid[:, -self.horizon:, :]

        trend_pred = self.trend_head(trend_future)
        residual_raw = self.residual_head(resid_future)

        bounded_residual = self.residual_output_scale * torch.tanh(residual_raw)
        return trend_pred + bounded_residual


def build_model(name: str, input_dim: int, horizon: int, cfg) -> nn.Module:
    name = name.lower()

    if name == "lstm":
        return LSTMForecast(input_dim, cfg.hidden_dim, cfg.n_layers, horizon, cfg.dropout)

    if name == "transformer":
        return TransformerForecast(
            input_dim,
            cfg.d_model,
            cfg.n_heads,
            cfg.n_layers,
            horizon,
            max_len=cfg.input_len + horizon + 8,
            dropout=cfg.dropout,
        )

    if name in (
        "stable-dr-cnn-transformer",
        "stable-stable-dr-cnn-transformer",
        "dr-cnn-transformer",
        "drcnntransformer",
        "dr_cnn_transformer",
        "cnn-transformer",
    ):
        return StableDRCNNTransformerForecast(
            input_dim=input_dim,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            horizon=horizon,
            kernel_size=cfg.cnn_kernel,
            decomp_kernel=cfg.get_decomp_kernel(horizon),
            residual_output_scale=cfg.get_residual_output_scale(horizon),
            max_len=cfg.input_len + horizon + 8,
            dropout=cfg.dropout,
        )

    raise ValueError(f"未知模型名: {name}")
