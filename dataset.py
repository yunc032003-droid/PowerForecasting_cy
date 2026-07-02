# -*- coding: utf-8 -*-
"""
dataset.py
==========
阶段二：滑窗构造 + 归一化 + DataLoader。

原则：
1) scaler 只 fit 自行划分后的 train 部分；
2) train/val 窗口只来自 train 部分；
3) test 目标区间必须完全落在 test 部分；
4) test 输入区间允许使用 train 末尾 90 天作为历史上下文。
"""

from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from config import Config
from data_preprocessing import FEATURE_COLS


FUTURE_KNOWN_COLS = [
    "historical_prior_target",
    "is_holiday",
    "is_before_holiday",
    "is_after_holiday",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "HDD",
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
]


def _split_train_val_starts(train_n: int, cfg: Config, horizon: int):
    all_starts = list(range(0, train_n - cfg.input_len - horizon + 1))
    if not all_starts:
        raise ValueError(
            f"train 数据长度 train_n={train_n} 不足以构造 horizon={horizon} 的训练窗口；"
            f"至少需要 input_len+horizon={cfg.input_len + horizon} 天。"
        )

    if len(all_starts) == 1:
        return all_starts, all_starts

    n_val = max(1, int(round(cfg.val_ratio * len(all_starts))))
    n_val = min(n_val, len(all_starts) - 1)

    train_starts = all_starts[:-n_val]
    val_starts = all_starts[-n_val:]
    return train_starts, val_starts


def _build_test_starts(train_n: int, full_n: int, cfg: Config, horizon: int):
    """
    full = train + test。
    测试目标必须完全在 test 区间：
        s + input_len >= train_n
        s + input_len + horizon <= full_n
    """
    s_min = max(0, train_n - cfg.input_len)
    s_max = full_n - cfg.input_len - horizon

    test_starts = [
        s for s in range(s_min, s_max + 1)
        if s + cfg.input_len >= train_n
    ]

    if not test_starts:
        test_n = full_n - train_n
        raise ValueError(
            f"test 数据长度 test_n={test_n} 不足以构造 horizon={horizon} 的测试窗口。"
            f"请把 config.py 中 test_days 设置为至少 {horizon}。"
        )
    return test_starts


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        starts,
        input_len: int,
        horizon: int,
        feature_cols: List[str],
        future_known_cols: List[str],
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.starts = list(starts)
        self.input_len = input_len
        self.horizon = horizon
        self.feature_cols = list(feature_cols)
        self.future_known_cols = list(future_known_cols)

        self.future_known_idx = [
            i for i, c in enumerate(self.feature_cols)
            if c in self.future_known_cols
        ]
        self.future_unknown_idx = [
            i for i, c in enumerate(self.feature_cols)
            if c not in self.future_known_cols
        ]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]

        x_past = self.X[s: s + self.input_len].copy()
        x_future = self.X[
            s + self.input_len:
            s + self.input_len + self.horizon
        ].copy()

        # 未来不可知特征在标准化空间置 0，即训练集均值。
        x_future[:, self.future_unknown_idx] = 0.0

        past_flag = np.zeros((self.input_len, 1), dtype=np.float32)
        future_flag = np.ones((self.horizon, 1), dtype=np.float32)

        x_past = np.concatenate([x_past, past_flag], axis=1)
        x_future = np.concatenate([x_future, future_flag], axis=1)
        x = np.concatenate([x_past, x_future], axis=0)

        y = self.y[
            s + self.input_len:
            s + self.input_len + self.horizon
        ]

        return torch.from_numpy(x), torch.from_numpy(y)


def build_loaders(train_table, test_table, cfg: Config, horizon: int) -> Tuple:
    train_table = train_table.sort_values("date").reset_index(drop=True)
    test_table = test_table.sort_values("date").reset_index(drop=True)
    full_df = pd.concat([train_table, test_table], ignore_index=True)

    train_n = len(train_table)
    full_n = len(full_df)

    train_starts, val_starts = _split_train_val_starts(train_n, cfg, horizon)
    test_starts = _build_test_starts(train_n, full_n, cfg, horizon)

    X_train_raw = train_table[FEATURE_COLS].values.astype(np.float64)
    y_train_raw = train_table[cfg.target_col].values.astype(np.float64)

    X_full_raw = full_df[FEATURE_COLS].values.astype(np.float64)
    y_full_raw = full_df[cfg.target_col].values.astype(np.float64)

    feat_scaler = StandardScaler().fit(X_train_raw)
    target_scaler = StandardScaler().fit(y_train_raw.reshape(-1, 1))

    X = feat_scaler.transform(X_full_raw).astype(np.float32)
    y = target_scaler.transform(y_full_raw.reshape(-1, 1)).ravel().astype(np.float32)

    future_known_cols = [c for c in FUTURE_KNOWN_COLS if c in FEATURE_COLS]

    train_ds = SlidingWindowDataset(
        X, y, train_starts,
        cfg.input_len, horizon,
        FEATURE_COLS, future_known_cols,
    )
    val_ds = SlidingWindowDataset(
        X, y, val_starts,
        cfg.input_len, horizon,
        FEATURE_COLS, future_known_cols,
    )
    test_ds = SlidingWindowDataset(
        X, y, test_starts,
        cfg.input_len, horizon,
        FEATURE_COLS, future_known_cols,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    feature_dim = len(FEATURE_COLS) + 1

    print(
        f"[loader] horizon={horizon} | train_windows={len(train_ds)} "
        f"val_windows={len(val_ds)} test_windows={len(test_ds)} | "
        f"train_days={train_n} test_days={len(test_table)}"
    )

    return train_loader, val_loader, test_loader, target_scaler, feature_dim
