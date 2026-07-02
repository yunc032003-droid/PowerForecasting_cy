# -*- coding: utf-8 -*-
"""
baselines.py
============
无需训练的对照方法，用于证明深度模型是否真正超过简单时序规则。

包含：
1. Seasonal Naive：去年同日值；
2. Historical Monthly Mean：历史同月均值；
3. Trend+Seasonal Decomposition：趋势 + 季节项，残差视为不可预测噪声。

所有 baseline 都使用 dataset.get_split_starts() 产生的同一组 test 窗口，
并在原始物理量纲下计算 MSE / MAE，与深度模型评估口径一致。
"""

import numpy as np
import pandas as pd

from dataset import get_split_starts


def _metrics(preds, trues):
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    return mse, mae


def seasonal_naive(table, cfg, horizon, test_starts):
    """
    Seasonal Naive:
        y_hat[t] = y[t - 365]

    如果预测窗口中某一天没有去年同期值，则退回到预测起点前一天的值。
    """
    y = table[cfg.target_col].values.astype(float)

    preds, trues = [], []
    for s in test_starts:
        origin = s + cfg.input_len
        target_idx = np.arange(origin, origin + horizon)

        pred = []
        for t in target_idx:
            lag = t - 365
            pred.append(y[lag] if lag >= 0 else y[origin - 1])

        preds.append(pred)
        trues.append(y[target_idx])

    preds = np.asarray(preds)
    trues = np.asarray(trues)
    mse, mae = _metrics(preds, trues)

    return {
        "mse": [mse],
        "mae": [mae],
        "last_pred": preds[-1],
        "last_true": trues[-1],
    }


def historical_monthly_mean(table, cfg, horizon, test_starts):
    """
    Historical Monthly Mean:
        对每个测试窗口，只使用预测起点 origin 之前的历史数据；
        预测未来某一天时，用历史同月平均用电量。
    """
    y = table[cfg.target_col].values.astype(float)
    dates = pd.to_datetime(table["date"])
    months = dates.dt.month.values

    preds, trues = [], []
    for s in test_starts:
        origin = s + cfg.input_len
        target_idx = np.arange(origin, origin + horizon)
        hist_idx = np.arange(0, origin)

        global_mean = y[hist_idx].mean()
        pred = []
        for t in target_idx:
            m = months[t]
            same_month_hist = hist_idx[months[hist_idx] == m]
            pred.append(y[same_month_hist].mean() if len(same_month_hist) else global_mean)

        preds.append(pred)
        trues.append(y[target_idx])

    preds = np.asarray(preds)
    trues = np.asarray(trues)
    mse, mae = _metrics(preds, trues)

    return {
        "mse": [mse],
        "mae": [mae],
        "last_pred": preds[-1],
        "last_true": trues[-1],
    }


def decomposition_baseline(table, cfg, horizon, test_starts, trend_window=30):
    """
    简单时序分解 baseline:
        y = trend + seasonality + residual

    预测时：
    - trend：使用预测起点前的滚动趋势并做简单线性外推；
    - seasonality：使用历史同 day-of-year 的去趋势残差均值；
    - residual：不预测，作为随机噪声讨论。

    注意：每个测试窗口只使用 origin 之前的信息，不使用目标区间真实值。
    """
    y = table[cfg.target_col].values.astype(float)
    dates = pd.to_datetime(table["date"])
    doy = dates.dt.dayofyear.values
    months = dates.dt.month.values

    # trailing rolling mean：trend[t] 只依赖 t 及其之前的历史值，不看未来。
    trend = (
        pd.Series(y)
        .rolling(trend_window, min_periods=7)
        .mean()
        .ffill()
        .bfill()
        .values
    )
    detrended = y - trend

    preds, trues = [], []
    for s in test_starts:
        origin = s + cfg.input_len
        target_idx = np.arange(origin, origin + horizon)
        hist_idx = np.arange(0, origin)

        last_trend = trend[origin - 1]

        # 用最近 60 天估计非常温和的趋势斜率。
        recent_start = max(0, origin - 60)
        recent = trend[recent_start:origin]
        if len(recent) >= 20:
            slope = (np.mean(recent[-10:]) - np.mean(recent[:10])) / max(1, len(recent))
        else:
            slope = 0.0

        pred = []
        for step, t in enumerate(target_idx):
            same_doy = hist_idx[doy[hist_idx] == doy[t]]
            if len(same_doy):
                seasonal = detrended[same_doy].mean()
            else:
                same_month = hist_idx[months[hist_idx] == months[t]]
                seasonal = detrended[same_month].mean() if len(same_month) else 0.0

            pred.append(last_trend + slope * (step + 1) + seasonal)

        preds.append(pred)
        trues.append(y[target_idx])

    preds = np.asarray(preds)
    trues = np.asarray(trues)
    mse, mae = _metrics(preds, trues)

    return {
        "mse": [mse],
        "mae": [mae],
        "last_pred": preds[-1],
        "last_true": trues[-1],
    }


def compute_baselines(table, cfg, horizon):
    """计算当前 horizon 的全部 baseline。"""
    N = len(table)
    _, _, test_starts = get_split_starts(N, cfg, horizon)

    return {
        "seasonal_naive": seasonal_naive(table, cfg, horizon, test_starts),
        "monthly_mean": historical_monthly_mean(table, cfg, horizon, test_starts),
        "decomposition": decomposition_baseline(table, cfg, horizon, test_starts),
    }
