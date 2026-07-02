# -*- coding: utf-8 -*-
"""
main.py
=======
7点优化版主控脚本。

新增：
1) 高清图与更好的边距；
2) 365天 clear MA 图；
3) 365天分段图；
4) 分段 MSE/MAE 分析；
5) 只保留 LSTM、Transformer、Stable-DR-CNN-Transformer 三条模型曲线。
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CFG
from data_preprocessing import build_train_test_feature_tables
from dataset import build_loaders
from train import train_one_run


MODEL_NAMES = ["LSTM", "Transformer", "Stable-DR-CNN-Transformer"]

MODEL_COLORS = {
    "LSTM": "#3A6EA5",
    "Transformer": "#F1A80A",
    "Stable-DR-CNN-Transformer": "#00A896",
}


def moving_average_1d(arr, window):
    arr = np.asarray(arr, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(arr, kernel, mode="valid")


def centered_ma_x(length, window):
    if window <= 1 or length < window:
        return np.arange(length)
    return np.arange(length - window + 1) + (window - 1) / 2.0


def make_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=CFG.figure_dpi, constrained_layout=True)
    return fig, ax


def setup_axes(ax, title, xlabel="Forecast Horizon (Days)", ylabel="Daily Summed Global Active Power"):
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    ax.grid(True, linestyle="-", color="#EAEAEA", linewidth=0.9)
    ax.tick_params(axis="both", labelsize=10)


def plot_core_models_raw(horizon_name, horizon_days, true_val, predictions_dict, pred_std_dict, output_dir="outputs"):
    is_long = horizon_days > 120
    fig, ax = make_fig((18, 7.6 if is_long else 7.0))

    gt_lw = 1.35 if is_long else 2.4
    pred_lw = 0.9 if is_long else 1.55

    x = np.arange(len(true_val))
    ax.plot(x, true_val, label="Ground Truth", color="black", linewidth=gt_lw, alpha=0.80, zorder=4)

    for name in MODEL_NAMES:
        pred = predictions_dict.get(name)
        if pred is None:
            continue

        ax.plot(
            x,
            pred,
            label=name,
            color=MODEL_COLORS.get(name),
            linewidth=pred_lw,
            alpha=0.86 if is_long else 0.96,
            zorder=3,
        )

        std = pred_std_dict.get(name)
        if std is not None and np.all(np.isfinite(std)):
            ax.fill_between(
                x,
                pred - std,
                pred + std,
                color=MODEL_COLORS.get(name),
                alpha=0.025 if is_long else 0.06,
                linewidth=0,
                zorder=1,
            )

    setup_axes(
        ax,
        f"Power Forecast vs Ground Truth — Core Models ({horizon_name.upper()}-term, {horizon_days} Days, 5-run Mean)",
        xlabel="Forecast Horizon (Days Relative to Start of Test Window)",
    )
    ax.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=0.93)
    ax.set_xlim(-2, horizon_days + 2)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"forecast_{horizon_name}_{horizon_days}d_core_models_5run_mean.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 原始三模型图已保存: {save_path}")


def plot_smoothed_models(horizon_name, horizon_days, true_val, predictions_dict, window, output_dir="outputs"):
    if len(true_val) < window:
        return

    fig, ax = make_fig((18 if horizon_days > 120 else 14, 7.0))

    x_ma = centered_ma_x(len(true_val), window)
    true_ma = moving_average_1d(true_val, window)

    ax.plot(x_ma, true_ma, label=f"Ground Truth (MA={window})", color="black", linewidth=2.4, zorder=5)

    for name in MODEL_NAMES:
        if name in predictions_dict:
            pred_ma = moving_average_1d(predictions_dict[name], window)
            ax.plot(
                x_ma,
                pred_ma,
                label=f"{name} (MA={window})",
                color=MODEL_COLORS.get(name),
                linewidth=1.8,
                alpha=0.96,
                zorder=4,
            )

    setup_axes(ax, f"MA={window} Smoothed Line Comparison ({horizon_name}-term, {horizon_days}d)")
    ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=0.95)
    ax.set_xlim(-2, horizon_days + 2)

    save_path = os.path.join(output_dir, f"forecast_{horizon_name}_{horizon_days}d_ma{window}_three_models.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] MA={window} 三模型平滑图已保存: {save_path}")


def plot_long_clear_smoothed(horizon_name, horizon_days, true_val, predictions_dict, output_dir="outputs"):
    if horizon_days <= 120:
        return

    window = int(CFG.long_clear_ma_window)
    fig, ax = make_fig((18, 7.5))

    x_ma = centered_ma_x(len(true_val), window)
    true_ma = moving_average_1d(true_val, window)

    ax.plot(x_ma, true_ma, label=f"Ground Truth (MA={window})", color="black", linewidth=2.7, alpha=0.92, zorder=5)

    for name in MODEL_NAMES:
        if name not in predictions_dict:
            continue
        pred_ma = moving_average_1d(predictions_dict[name], window)
        ax.plot(
            x_ma,
            pred_ma,
            label=f"{name} (MA={window})",
            color=MODEL_COLORS.get(name),
            linewidth=2.0,
            alpha=0.95,
            zorder=4,
        )

    setup_axes(ax, f"Clear Long-term Forecast Comparison — MA={window} ({horizon_days} Days, 5-run Mean)")
    ax.legend(loc="upper right", fontsize=10, frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=0.95)
    ax.set_xlim(-2, horizon_days + 2)

    save_path = os.path.join(output_dir, f"forecast_{horizon_name}_{horizon_days}d_clear_ma{window}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 365天高清平滑图已保存: {save_path}")


def plot_long_segments(horizon_name, horizon_days, true_val, predictions_dict, output_dir="outputs"):
    if horizon_days <= 120:
        return

    seg = int(CFG.long_segment_days)
    n = len(true_val)
    num_segments = int(math.ceil(n / seg))

    for idx in range(num_segments):
        start = idx * seg
        end = min((idx + 1) * seg, n)
        if end - start < 10:
            continue

        fig, ax = make_fig((14, 6.6))
        x = np.arange(start, end)

        ax.plot(x, true_val[start:end], label="Ground Truth", color="black", linewidth=2.1, alpha=0.86, zorder=4)

        for name in MODEL_NAMES:
            if name in predictions_dict:
                ax.plot(
                    x,
                    predictions_dict[name][start:end],
                    label=name,
                    color=MODEL_COLORS.get(name),
                    linewidth=1.45,
                    alpha=0.93,
                    zorder=3,
                )

        setup_axes(ax, f"Long-term Forecast Segment {idx + 1}: Days {start}-{end - 1}")
        ax.legend(loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="#DDDDDD", framealpha=0.95)
        ax.set_xlim(start - 1, end)

        save_path = os.path.join(output_dir, f"forecast_{horizon_name}_{horizon_days}d_segment_{idx + 1}_{start}_{end - 1}.png")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] 365天分段高清图已保存: {save_path}")


def compute_segment_errors(horizon_name, horizon_days, true_curve, predictions_dict):
    records = []
    seg = int(CFG.segment_error_days if horizon_days > 120 else 30)
    n = len(true_curve)
    for name, pred in predictions_dict.items():
        for start in range(0, n, seg):
            end = min(start + seg, n)
            if end - start < 2:
                continue
            diff = pred[start:end] - true_curve[start:end]
            records.append({
                "Horizon": f"{horizon_name}({horizon_days})",
                "Model": name,
                "Segment": f"{start}-{end - 1}",
                "StartDay": start,
                "EndDay": end - 1,
                "MSE": float(np.mean(diff ** 2)),
                "MAE": float(np.mean(np.abs(diff))),
            })
    return records


def run_model_variant(model_name, loaders, target_scaler, feature_dim, horizon_days, cfg):
    run_mses, run_maes, run_preds, run_trues = [], [], [], []

    for run_idx in range(1, cfg.num_runs + 1):
        seed = cfg.base_seed + run_idx * 100
        res = train_one_run(
            model_name=model_name,
            loaders=(loaders[0], loaders[1], loaders[2]),
            target_scaler=target_scaler,
            feature_dim=feature_dim,
            horizon=horizon_days,
            cfg=cfg,
            seed=seed,
        )
        run_mses.append(res["mse"])
        run_maes.append(res["mae"])
        run_preds.append(res["preds"])
        run_trues.append(res["trues"])

    pred_stack = np.stack(run_preds, axis=0)
    mean_preds = np.mean(pred_stack, axis=0)
    std_preds = np.std(pred_stack, axis=0)

    return {
        "model_name": model_name,
        "mse_mean": float(np.mean(run_mses)),
        "mse_std": float(np.std(run_mses)),
        "mae_mean": float(np.mean(run_maes)),
        "mae_std": float(np.std(run_maes)),
        "mean_preds": mean_preds,
        "std_preds": std_preds,
        "trues": run_trues[0],
    }


def main():
    print("=" * 88)
    print(" 家庭电力预测 7点优化版：滚动特征 + 分任务超参 + 趋势主导残差辅助输出")
    print("=" * 88)

    CFG.apply_quick_test()
    os.makedirs(CFG.output_dir, exist_ok=True)

    print("\n[Step 1] 从 household_power_consumption.txt 构建并自行划分 train/test 日级特征表...")
    train_table, test_table = build_train_test_feature_tables(CFG, save=True)

    summary_records = []
    segment_records = []

    for horizon_name, horizon_days in CFG.horizons.items():
        print("\n" + "=" * 88)
        print(f" 子任务：{horizon_name.upper()} 预测 | horizon = {horizon_days} 天")
        print("=" * 88)

        print(
            f"[task params] decomp_kernel={CFG.get_decomp_kernel(horizon_days)}, "
            f"peak_beta={CFG.get_peak_weight_beta(horizon_days)}, "
            f"diff_gamma={CFG.get_diff_loss_gamma(horizon_days)}, "
            f"residual_scale={CFG.get_residual_output_scale(horizon_days)}"
        )

        loaders = build_loaders(train_table, test_table, CFG, horizon=horizon_days)
        train_loader, val_loader, test_loader, target_scaler, feature_dim = loaders

        plot_preds = {}
        plot_stds = {}
        ground_truth_curve = None

        for model_name in MODEL_NAMES:
            print(f"\n  --> 模型 {model_name} 开始 {CFG.num_runs} 轮独立实验")
            result = run_model_variant(
                model_name=model_name,
                loaders=loaders,
                target_scaler=target_scaler,
                feature_dim=feature_dim,
                horizon_days=horizon_days,
                cfg=CFG,
            )

            plot_index = -1
            plot_preds[model_name] = result["mean_preds"][plot_index]
            plot_stds[model_name] = result["std_preds"][plot_index]

            if ground_truth_curve is None:
                ground_truth_curve = result["trues"][plot_index]

            summary_records.append({
                "Horizon": f"{horizon_name}({horizon_days})",
                "Model": model_name,
                "MSE": f"{result['mse_mean']:.3f} ± {result['mse_std']:.3f}",
                "MAE": f"{result['mae_mean']:.3f} ± {result['mae_std']:.3f}",
            })

            print(
                f"  ==> {model_name}: "
                f"MSE={result['mse_mean']:.3f}±{result['mse_std']:.3f} | "
                f"MAE={result['mae_mean']:.3f}±{result['mae_std']:.3f}"
            )

        segment_records.extend(compute_segment_errors(horizon_name, horizon_days, ground_truth_curve, plot_preds))

        print("\n  --> 导出三模型预测曲线...")
        plot_core_models_raw(horizon_name, horizon_days, ground_truth_curve, plot_preds, plot_stds, CFG.output_dir)

        for w in CFG.report_ma_windows:
            plot_smoothed_models(horizon_name, horizon_days, ground_truth_curve, plot_preds, w, CFG.output_dir)

        plot_long_clear_smoothed(horizon_name, horizon_days, ground_truth_curve, plot_preds, CFG.output_dir)
        plot_long_segments(horizon_name, horizon_days, ground_truth_curve, plot_preds, CFG.output_dir)

    df_summary = pd.DataFrame(summary_records)
    df_segments = pd.DataFrame(segment_records)

    summary_csv = os.path.join(CFG.output_dir, "results_summary.csv")
    summary_txt = os.path.join(CFG.output_dir, "results_summary.txt")
    segment_csv = os.path.join(CFG.output_dir, "segment_errors.csv")
    segment_txt = os.path.join(CFG.output_dir, "segment_errors.txt")

    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    df_segments.to_csv(segment_csv, index=False, encoding="utf-8-sig")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("机器学习课程考核实验矩阵：LSTM / Transformer / Stable-DR-CNN-Transformer Final，mean±std\n")
        f.write("=" * 100 + "\n")
        f.write(df_summary.to_string(index=False))
        f.write("\n\n")
        f.write("7点优化说明：\n")
        f.write("1. 使用 constrained_layout 与长周期清晰图、分段图改善可读性。\n")
        f.write("2. 创新损失采用更稳健的峰谷加权 Huber + 差分一致性。\n")
        f.write("3. 趋势-残差分解窗口按任务区分：短期 15，长期 30。\n")
        f.write("4. 加入 lag 与 rolling mean/std/min/max 滚动统计特征。\n")
        f.write("5. 90天和365天使用不同 peak/diff/residual 超参数。\n")
        f.write("6. 输出 segment_errors.csv 分析不同预测阶段误差。\n")
        f.write("7. 创新模型显式输出 trend_pred + bounded_residual_pred（残差轻量限幅）。\n")

    with open(segment_txt, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("分段误差分析：基于最终绘图窗口的多轮平均预测曲线\n")
        f.write("=" * 100 + "\n")
        f.write(df_segments.to_string(index=False))

    print("\n" + open(summary_txt, "r", encoding="utf-8").read())
    print(f"[done] 全部结果已保存到: {CFG.output_dir}")


if __name__ == "__main__":
    main()
