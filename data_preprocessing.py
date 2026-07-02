# -*- coding: utf-8 -*-
"""
data_preprocessing.py
=====================
阶段一：数据预处理与特征工程。

适配场景：
1) 只读取 household_power_consumption.txt；
2) 按天聚合后，按时间顺序自行划分 train/test；
3) test 默认取最后 365 天，保证 long horizon=365 可评估；
4) historical_prior_target 使用 shift(365).ffill()，不再 bfill；
5) 前 365 天缺失值只用训练集月份均值/训练集整体均值填充，避免未来信息泄漏。
"""

import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import Config


FEATURE_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "sub_metering_remainder",
    "historical_prior_target",
    "RR",
    "NBJRR1",
    "NBJRR5",
    "NBJRR10",
    "NBJBROU",
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
]

ROLLING_FEATURE_COLS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_30",
    "rolling_min_7",
    "rolling_max_7",
]

FEATURE_COLS = FEATURE_COLS + ROLLING_FEATURE_COLS


PARIS_MONTHLY_TMEAN = {
    1: 4.9, 2: 5.6, 3: 8.6, 4: 11.2, 5: 14.9, 6: 17.8,
    7: 20.1, 8: 19.9, 9: 16.8, 10: 12.8, 11: 8.0, 12: 5.5,
}


def _first_existing(data_dir: str, candidates: List[str]) -> str:
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    return ""


def _read_table_auto(path: str) -> pd.DataFrame:
    last_err = None
    for kwargs in [
        {"sep": ";"},
        {"sep": ","},
        {"sep": "\t"},
        {"sep": None, "engine": "python"},
    ]:
        try:
            df = pd.read_csv(path, na_values="?", low_memory=False, **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"无法读取文件 {path}: {last_err}")


def _col_key(c: str) -> str:
    return str(c).strip().lower().replace("_", "").replace(" ", "").replace("-", "")


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    alias = {
        "date": "Date",
        "time": "Time",
        "datetime": "datetime",
        "timestamp": "datetime",
        "globalactivepower": "Global_active_power",
        "globalreactivepower": "Global_reactive_power",
        "voltage": "Voltage",
        "globalintensity": "Global_intensity",
        "submetering1": "Sub_metering_1",
        "submetering2": "Sub_metering_2",
        "submetering3": "Sub_metering_3",
    }
    rename = {}
    for c in df.columns:
        k = _col_key(c)
        if k in alias:
            rename[c] = alias[k]
    return df.rename(columns=rename)


def load_power_raw(cfg: Config) -> pd.DataFrame:
    paths = [
        os.path.join(cfg.data_dir, f)
        for f in cfg.power_files
        if os.path.exists(os.path.join(cfg.data_dir, f))
    ]

    if not paths:
        fallback = os.path.join(cfg.data_dir, cfg.power_fallback_file)
        if os.path.exists(fallback):
            paths = [fallback]
            print(f"[power] 未找到 household_power_consumption.txt，回退使用: {fallback}")
        else:
            raise FileNotFoundError(
                f"未找到电力数据文件。请把 household_power_consumption.txt 放到 data_dir={cfg.data_dir!r}。"
            )

    frames = []
    for p in paths:
        df = _read_table_auto(p)
        df = _canonicalize_columns(df)
        frames.append(df)
        print(f"[power] 读取 {p} -> {df.shape}")

    raw = pd.concat(frames, ignore_index=True)

    if "datetime" in raw.columns:
        dt = pd.to_datetime(raw["datetime"], errors="coerce", dayfirst=True)
    elif "Date" in raw.columns and "Time" in raw.columns:
        dt = pd.to_datetime(
            raw["Date"].astype(str) + " " + raw["Time"].astype(str),
            format="%d/%m/%Y %H:%M:%S",
            errors="coerce",
        )
        if dt.notna().mean() < 0.8:
            dt = pd.to_datetime(
                raw["Date"].astype(str) + " " + raw["Time"].astype(str),
                errors="coerce",
                dayfirst=True,
            )
    elif "Date" in raw.columns:
        dt = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True)
    else:
        raise ValueError("电力数据中没有找到 Date/Time 或 datetime 列。")

    raw["datetime"] = dt
    raw = raw.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    num_cols = [
        "Global_active_power", "Global_reactive_power", "Voltage",
        "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
    ]

    if "Global_active_power" not in raw.columns:
        raise ValueError("缺少目标列 Global_active_power。")

    for c in num_cols:
        if c not in raw.columns:
            raw[c] = 0.0
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw[num_cols] = raw[num_cols].ffill().bfill()
    return raw


def aggregate_daily(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["date"] = raw["datetime"].dt.floor("D")

    agg_map = {
        "Global_active_power": "sum",
        "Global_reactive_power": "sum",
        "Sub_metering_1": "sum",
        "Sub_metering_2": "sum",
        "Sub_metering_3": "sum",
        "Voltage": "mean",
        "Global_intensity": "mean",
    }
    daily = raw.groupby("date").agg(agg_map).reset_index()

    daily["sub_metering_remainder"] = (
        daily["Global_active_power"] * 1000.0 / 60.0
        - (daily["Sub_metering_1"] + daily["Sub_metering_2"] + daily["Sub_metering_3"])
    )

    daily["YYYYMM"] = daily["date"].dt.year * 100 + daily["date"].dt.month
    daily = daily.sort_values("date").reset_index(drop=True)

    print(
        f"[power] 按天聚合完成 -> {daily.shape}，日期范围 "
        f"{daily['date'].min().date()} ~ {daily['date'].max().date()}"
    )
    return daily


def load_weather_monthly(cfg: Config) -> pd.DataFrame:
    path = _first_existing(cfg.data_dir, cfg.weather_files)
    if not path:
        warnings.warn("未找到天气文件，将以全 0 天气特征继续。")
        return pd.DataFrame(
            columns=["YYYYMM", "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
        )

    w = _read_table_auto(path)
    print(f"[weather] 读取 {path} -> {w.shape}")

    rename = {}
    for c in w.columns:
        k = _col_key(c)
        if k == "numposte":
            rename[c] = "NUM_POSTE"
        elif k in ("aaaamm", "yyyymm"):
            rename[c] = "AAAAMM"
        elif k in ("date", "datetime"):
            rename[c] = "date"
        elif k == "rr":
            rename[c] = "RR"
        elif k == "nbjrr1":
            rename[c] = "NBJRR1"
        elif k == "nbjrr5":
            rename[c] = "NBJRR5"
        elif k == "nbjrr10":
            rename[c] = "NBJRR10"
        elif k == "nbjbrou":
            rename[c] = "NBJBROU"
    w = w.rename(columns=rename)

    if "AAAAMM" not in w.columns and "date" in w.columns:
        dt = pd.to_datetime(w["date"], errors="coerce", dayfirst=True)
        w["AAAAMM"] = dt.dt.year * 100 + dt.dt.month

    keep = ["NUM_POSTE", "AAAAMM", "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
    keep = [c for c in keep if c in w.columns]
    w = w[keep].copy()

    if "AAAAMM" not in w.columns:
        warnings.warn("天气文件没有 AAAAMM/YYYYMM/date，天气特征将置 0。")
        return pd.DataFrame(
            columns=["YYYYMM", "RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
        )

    if "NUM_POSTE" in w.columns:
        candidates = [cfg.preferred_station] + list(cfg.fallback_stations)
        chosen = None
        poste_num = pd.to_numeric(w["NUM_POSTE"], errors="coerce")
        for sid in candidates:
            if (poste_num == sid).any():
                chosen = sid
                break

        if chosen is not None:
            w = w[poste_num == chosen].copy()
            print(f"[weather] 使用站点 NUM_POSTE={chosen}")
        else:
            print("[weather] 候选站点均不存在，改用全部站点月度均值。")
            w = w.groupby("AAAAMM", as_index=False).mean(numeric_only=True)

    w = w.rename(columns={"AAAAMM": "YYYYMM"})

    if "RR" in w.columns and cfg.rr_divide_by_10:
        w["RR"] = pd.to_numeric(w["RR"], errors="coerce") / 10.0

    weather_cols = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
    for c in weather_cols:
        if c not in w.columns:
            w[c] = 0.0
        w[c] = pd.to_numeric(w[c], errors="coerce")

    return w[["YYYYMM"] + weather_cols].drop_duplicates("YYYYMM")


def _easter_sunday(year: int):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(year=year, month=month, day=day)


def _french_holidays_fallback(years: List[int]) -> set:
    hol = set()
    for y in years:
        easter = _easter_sunday(y)
        movable = [
            easter + pd.Timedelta(days=1),
            easter + pd.Timedelta(days=39),
            easter + pd.Timedelta(days=50),
        ]
        fixed = [
            pd.Timestamp(y, 1, 1),
            pd.Timestamp(y, 5, 1),
            pd.Timestamp(y, 5, 8),
            pd.Timestamp(y, 7, 14),
            pd.Timestamp(y, 8, 15),
            pd.Timestamp(y, 11, 1),
            pd.Timestamp(y, 11, 11),
            pd.Timestamp(y, 12, 25),
        ]
        for d in movable + fixed:
            hol.add(pd.Timestamp(d).normalize())
    return hol


def _build_holiday_set(years: List[int]) -> set:
    try:
        import holidays as _h
        fr = _h.France(years=years)
        return {pd.Timestamp(d).normalize() for d in fr.keys()}
    except Exception as e:
        print(f"[holiday] 未能加载 holidays 库（{e}），使用内置法国节假日。")
        return _french_holidays_fallback(years)


def _choose_train_test_split(daily: pd.DataFrame, cfg: Config) -> int:
    n = len(daily)
    max_horizon = max(int(v) for v in cfg.horizons.values())
    test_days = max(int(cfg.test_days), max_horizon)

    min_train_days = cfg.input_len + max_horizon + 10
    if n - test_days < min_train_days:
        raise ValueError(
            f"数据天数 N={n} 不足以按 test_days={test_days} 划分。"
            f"至少需要 train 部分 {min_train_days} 天 + test 部分 {test_days} 天。"
        )

    train_n = n - test_days
    print(
        f"[split] 自行按时间顺序划分：train/val 前 {train_n} 天，"
        f"test 最后 {test_days} 天。"
    )
    return train_n



def _fill_by_train_month_or_global(df: pd.DataFrame, col: str, train_mask: pd.Series, default: float = 0.0):
    month = df["date"].dt.month
    train_values = df.loc[train_mask, col]
    if train_values.notna().any():
        month_mean = df.loc[train_mask].groupby(df.loc[train_mask, "date"].dt.month)[col].mean()
        global_mean = float(train_values.mean())
    else:
        month_mean = pd.Series(dtype=float)
        global_mean = default
    fill_by_month = month.map(month_mean).astype(float).fillna(global_mean)
    df[col] = df[col].fillna(fill_by_month)
    return df


def add_rolling_features(df: pd.DataFrame, train_mask: pd.Series, cfg: Config) -> pd.DataFrame:
    """增加滞后与滚动统计特征。所有 rolling 都基于 shift(1)，只使用过去信息。"""
    df = df.copy()
    g = df["Global_active_power"].astype(float)
    shifted = g.shift(1)

    for lag in cfg.lag_days:
        df[f"lag_{lag}"] = g.shift(lag)

    for w in cfg.rolling_windows:
        df[f"rolling_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()

    df["rolling_std_7"] = shifted.rolling(7, min_periods=2).std()
    df["rolling_std_30"] = shifted.rolling(30, min_periods=2).std()
    df["rolling_min_7"] = shifted.rolling(7, min_periods=1).min()
    df["rolling_max_7"] = shifted.rolling(7, min_periods=1).max()

    train_global = float(df.loc[train_mask, "Global_active_power"].mean())
    for c in ROLLING_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
        if "std" in c:
            train_std_mean = df.loc[train_mask, c].mean()
            if pd.isna(train_std_mean):
                train_std_mean = 0.0
            df[c] = df[c].fillna(train_std_mean)
        else:
            df = _fill_by_train_month_or_global(df, c, train_mask, default=train_global)
    return df


def add_advanced_features(df: pd.DataFrame, train_mask: pd.Series, cfg: Config) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    train_mask = pd.Series(train_mask).reset_index(drop=True).astype(bool)

    # 只使用过去信息，严禁 bfill。
    df["historical_prior_target"] = df["Global_active_power"].shift(365).ffill()
    train_global_mean = float(df.loc[train_mask, "Global_active_power"].mean())
    df = _fill_by_train_month_or_global(df, "historical_prior_target", train_mask, default=train_global_mean)

    years = sorted(df["date"].dt.year.unique().tolist())
    years_for_holidays = list(range(min(years) - 1, max(years) + 2))
    holiday_set = _build_holiday_set(years_for_holidays)

    norm_date = df["date"].dt.normalize()
    df["is_holiday"] = norm_date.isin(holiday_set).astype(int)
    df["is_before_holiday"] = (norm_date + pd.Timedelta(days=1)).isin(holiday_set).astype(int)
    df["is_after_holiday"] = (norm_date - pd.Timedelta(days=1)).isin(holiday_set).astype(int)

    dow = df["date"].dt.dayofweek
    month = df["date"].dt.month
    doy = df["date"].dt.dayofyear

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    tmean = month.map(PARIS_MONTHLY_TMEAN).astype(float)
    df["HDD"] = (18.0 - tmean).clip(lower=0.0)

    if cfg.use_rolling_features:
        df = add_rolling_features(df, train_mask, cfg)

    return df


def build_train_test_feature_tables(cfg: Config, save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_power_raw(cfg)
    daily = aggregate_daily(raw)

    train_n = _choose_train_test_split(daily, cfg)
    daily["split"] = "train"
    daily.loc[train_n:, "split"] = "test"

    weather = load_weather_monthly(cfg)
    merged = daily.merge(weather, on="YYYYMM", how="left")

    for c in ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

        train_weather_mean = merged.loc[merged["split"] == "train", c].mean()
        if pd.isna(train_weather_mean):
            train_weather_mean = 0.0
        merged[c] = merged[c].fillna(train_weather_mean)

    train_mask = merged["split"].eq("train")
    merged = add_advanced_features(merged, train_mask=train_mask, cfg=cfg)

    for c in FEATURE_COLS:
        if c not in merged.columns:
            merged[c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)

    out_cols = ["date", "split"] + FEATURE_COLS
    merged = merged[out_cols].sort_values("date").reset_index(drop=True)

    train_table = merged[merged["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    test_table = merged[merged["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    if save:
        os.makedirs(cfg.output_dir, exist_ok=True)

        all_path = os.path.join(cfg.output_dir, cfg.processed_all_csv)
        train_path = os.path.join(cfg.output_dir, cfg.processed_train_csv)
        test_path = os.path.join(cfg.output_dir, cfg.processed_test_csv)

        merged.to_csv(all_path, index=False, encoding="utf-8-sig")
        train_table.to_csv(train_path, index=False, encoding="utf-8-sig")
        test_table.to_csv(test_path, index=False, encoding="utf-8-sig")

        print(f"[save] 全量日级特征表 -> {all_path} shape={merged.shape}")
        print(f"[save] train 日级特征表 -> {train_path} shape={train_table.shape}")
        print(f"[save] test  日级特征表 -> {test_path} shape={test_table.shape}")

    return train_table, test_table


def build_daily_feature_table(cfg: Config, save: bool = True) -> pd.DataFrame:
    train_table, test_table = build_train_test_feature_tables(cfg, save=save)
    return pd.concat([train_table, test_table], ignore_index=True)
