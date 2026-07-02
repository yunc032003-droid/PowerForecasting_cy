# -*- coding: utf-8 -*-
"""
config.py
=========
最终稳定版配置。

设计目标：
1. 只使用 household_power_consumption.txt，并按时间顺序自行划分 train/test；
2. 只比较 LSTM、Transformer、Stable-DR-CNN-Transformer 三类模型；
3. 创新模型采用“趋势主导、残差辅助”策略，避免长期预测过度震荡；
4. 创新损失以 MSE 为主，附加小权重峰谷项、差分项、趋势一致性项；
5. 保留滚动统计特征、365天高清平滑图、365天分段图、分段误差表。
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ------------------------------------------------------------------
    # 1. 路径
    # ------------------------------------------------------------------
    data_dir: str = "."
    power_files: List[str] = field(default_factory=lambda: ["household_power_consumption.txt"])
    power_fallback_file: str = "data.txt"
    weather_files: List[str] = field(default_factory=lambda: ["weather.txt", "weather_2006_2011.csv"])

    output_dir: str = "outputs"
    processed_train_csv: str = "daily_features_train_self_split.csv"
    processed_test_csv: str = "daily_features_test_self_split.csv"
    processed_all_csv: str = "daily_features_all_self_split.csv"

    # ------------------------------------------------------------------
    # 2. 数据划分
    # ------------------------------------------------------------------
    test_days: int = 365
    val_ratio: float = 0.1

    # ------------------------------------------------------------------
    # 3. 天气
    # ------------------------------------------------------------------
    preferred_station: int = 94054001
    fallback_stations: List[int] = field(default_factory=lambda: [92007001, 92048001, 92025001])
    rr_divide_by_10: bool = False

    # ------------------------------------------------------------------
    # 4. 预测任务
    # ------------------------------------------------------------------
    input_len: int = 90
    horizons: dict = field(default_factory=lambda: {"short": 90, "long": 365})
    target_col: str = "Global_active_power"

    # ------------------------------------------------------------------
    # 5. 训练
    # ------------------------------------------------------------------
    batch_size: int = 16
    epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 1e-5
    num_runs: int = 5
    base_seed: int = 2024
    early_stop_patience: int = 20
    grad_clip: float = 1.0

    # ------------------------------------------------------------------
    # 6. 基础模型维度
    # ------------------------------------------------------------------
    hidden_dim: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    cnn_kernel: int = 3

    # ------------------------------------------------------------------
    # 7. 创新模型：分任务稳定超参数
    # ------------------------------------------------------------------
    # 90天：保留一定残差响应；365天：趋势主导，残差显著收敛。
    decomp_kernel_short: int = 15
    decomp_kernel_long: int = 30

    residual_output_scale_short: float = 0.50
    residual_output_scale_long: float = 0.20

    # 损失尽量靠近最终评价指标 MSE/MAE，特殊项只作为轻微正则。
    peak_weight_beta_short: float = 0.05
    peak_weight_beta_long: float = 0.03

    diff_loss_gamma_short: float = 0.02
    diff_loss_gamma_long: float = 0.01

    trend_loss_lambda_short: float = 0.05
    trend_loss_lambda_long: float = 0.08

    peak_weight_max: float = 1.8
    huber_beta: float = 0.6
    trend_ma_window_short: int = 7
    trend_ma_window_long: int = 14

    # ------------------------------------------------------------------
    # 8. 滚动统计特征
    # ------------------------------------------------------------------
    use_rolling_features: bool = True
    lag_days: List[int] = field(default_factory=lambda: [1, 7, 14])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])

    # ------------------------------------------------------------------
    # 9. 绘图与分段误差
    # ------------------------------------------------------------------
    figure_dpi: int = 300
    report_ma_windows: List[int] = field(default_factory=lambda: [7, 30])
    long_clear_ma_window: int = 14
    long_segment_days: int = 90
    segment_error_days: int = 90

    # ------------------------------------------------------------------
    # 10. 运行控制
    # ------------------------------------------------------------------
    device: str = "cuda"
    quick_test: bool = False
    num_workers: int = 0

    def get_decomp_kernel(self, horizon: int) -> int:
        return self.decomp_kernel_long if horizon > 100 else self.decomp_kernel_short

    def get_residual_output_scale(self, horizon: int) -> float:
        return self.residual_output_scale_long if horizon > 100 else self.residual_output_scale_short

    def get_peak_weight_beta(self, horizon: int) -> float:
        return self.peak_weight_beta_long if horizon > 100 else self.peak_weight_beta_short

    def get_diff_loss_gamma(self, horizon: int) -> float:
        return self.diff_loss_gamma_long if horizon > 100 else self.diff_loss_gamma_short

    def get_trend_loss_lambda(self, horizon: int) -> float:
        return self.trend_loss_lambda_long if horizon > 100 else self.trend_loss_lambda_short

    def get_trend_ma_window(self, horizon: int) -> int:
        return self.trend_ma_window_long if horizon > 100 else self.trend_ma_window_short

    def apply_quick_test(self):
        if self.quick_test:
            self.epochs = 3
            self.num_runs = 2
            self.early_stop_patience = 2
            print("[Config] quick_test=True -> epochs=3, num_runs=2")


CFG = Config()
