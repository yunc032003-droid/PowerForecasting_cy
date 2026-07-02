# 家庭电力消耗多变量时间序列预测（LSTM / Transformer / CNN-Transformer）

本项目面向 **2026 专硕机器学习课程项目**，基于 UCI *Individual household electric power
consumption* 数据集与配套的法国月度气象数据，完成对家庭**日总有功功率
（Global_active_power）**的短期（90 天）与长期（365 天）多变量时间序列预测。

实现包含三种模型并按统一规范评估：

1. **LSTM** —— 经典多层 LSTM + MLP 头；
2. **Transformer** —— 位置编码 + 标准 Transformer Encoder；
3. **CNN-Transformer（创新改进模型）** —— Conv1d 提取局部时序/多变量交叉特征，
   再接 Transformer Encoder 建模长程依赖。

短期与长期模型**完全独立训练**（长期模型参数不用于短期）。每个模型在每个 horizon 上
进行 **5 轮不同随机种子实验**，报告 **MSE / MAE 的 Mean ± Std**，并绘制最后一轮测试集上
**预测值 vs Ground Truth** 的对比曲线。

---

## 一、环境依赖

```bash
pip install -r requirements.txt
```

| 库 | 作用 |
|---|---|
| `torch>=1.12` | 模型构建与训练（`TransformerEncoderLayer` 需 `batch_first` 支持） |
| `pandas`, `numpy` | 数据读取、按天聚合、特征工程 |
| `scikit-learn` | `StandardScaler` 归一化 |
| `holidays` | 自动生成法国法定节假日标签 |
| `matplotlib` | 绘制预测对比图 |

> **离线兜底**：若 `holidays` 库不可用，代码会自动回退到内置的法国法定节假日算法
> （固定节日 + 复活节派生节日），保证离线也能运行，结果一致。

GPU 可选：`config.py` 中 `device="cuda"`，无 GPU 时自动回退 CPU。

---

## 二、数据放置规范

```
power-forecasting/
├── train.csv            # 课程提供的电力训练数据（分钟级，';' 分隔，'?' 为缺失）
├── test.csv             # 课程提供的电力测试数据
├── weather.txt          # 气象数据（月度，';' 分隔）；也支持 weather_2006_2011.csv
├── config.py
├── data_preprocessing.py
├── dataset.py
├── models.py
├── train.py
├── main.py
└── outputs/             # 运行后自动生成：中间特征表、结果汇总、对比图
```

- 电力数据：把课程的 `train.csv`、`test.csv` 放在项目根目录即可（文件名在
  `config.power_files` 中配置）。代码会把二者拼接成完整时间轴，再做按天聚合与特征工程。
  若暂时只有样例 `data.txt`，会自动回退使用它（但样例仅含单日，不足以训练，
  请务必使用完整数据）。
- 天气数据：放 `weather.txt` 或 `weather_2006_2011.csv`，两种文件名都会自动尝试
  （见 `config.weather_files`）。

### ⚠️ 关于天气站点（重要）

指导书示例站点为 **94054001（巴黎奥利机场）**，但课程实际下发的天气文件中
**不包含该站点**，而是包含 Hauts-de-Seine（92 省）若干站点
（BAGNEUX / COLOMBES / COURBEVOIE / FONTENAY / MEUDON / SURESNES）。
其中 **BAGNEUX(92007001) 紧邻数据采集地 Sceaux（索镇）**，是更优的地理代理。

代码逻辑（`config.py`）：
1. 优先匹配 `preferred_station`（默认 94054001）；
2. 不存在则按 `fallback_stations`（默认 `[92007001, 92048001, 92025001]`）顺序匹配；
3. 仍不存在则对当月所有站点取平均。

如需强制指定站点，直接修改 `config.preferred_station` 即可。

### 关于 `RR` 单位

指导书称 `RR` 为“毫米的十分之一，需除以 10”。但本数据文件中的 `RR` 实际已是毫米
（月累计 ~50mm，符合巴黎气候）。因此 `config.rr_divide_by_10` 默认为 `False`。
无论是否除以 10，`StandardScaler` 都会将其归一化抵消，不影响建模结果。

---

## 三、执行步骤与整体流程

### 1. 一键运行完整实验

```bash
python main.py
```

执行流水线（对应代码阶段）：

1. **阶段一 · 数据预处理与特征工程**（`data_preprocessing.py`）
   - 分钟级电力数据：`'?' → NaN`，前向填充；
   - **按天聚合（严格对齐指导书）**：
     - `Global_active_power / Global_reactive_power / Sub_metering_1 / Sub_metering_2 / Sub_metering_3` → **求和**；
     - `Voltage / Global_intensity` → **求平均**；
   - 计算隐含特征 `sub_metering_remainder = (GAP×1000/60) − (sub1+sub2+sub3)`；
   - 月级天气按 `YYYYMM` **左连接**到每一天；
   - **三大高级特征**：法国法定节假日 `is_holiday`、时间周期正余弦编码
     （`dow_sin/cos`、`month_sin/cos`）、采暖度日数 `HDD = max(0, 18 − T_month)`
     （巴黎月度常年均温作为物理先验）；
   - 输出天级特征表 `outputs/daily_features.csv`（共 **19 个特征列**）。

2. **阶段二 · 滑窗与归一化**（`dataset.py`）
   - `StandardScaler` 仅在训练时间段 `fit`，再 `transform` 全序列（避免泄漏）；
   - 输入窗口 `90` 天，输出 `90`（短期）/ `365`（长期）天；
   - 样本形状：`x:(B,90,19)`，`y:(B,horizon)`。

3. **阶段三 · 三大模型**（`models.py`）：LSTM / Transformer / CNN-Transformer。

4. **阶段四 · 5 轮实验 + 评估 + 可视化**（`train.py` + `main.py`）
   - 外层循环 5 轮（种子 `base_seed + k`）；
   - 训练用 MSE 损失，早停用 **val 集**（取自训练段尾部，不窥探 test）；
   - 测试阶段对预测/真实值 `inverse_transform` 反归一化，在原始量纲下算 MSE / MAE；
   - 打印 `Mean ± Std` 结果表（`outputs/results_summary.txt`）；
   - 绘制对比图：`outputs/forecast_short_90d.png`、`outputs/forecast_long_365d.png`。

### 2. 快速冒烟测试（先跑通再正式跑）

```bash
python main.py --quick          # epochs=3, runs=2，CPU 上数分钟内跑通
```

### 3. 配置长短期参数 / 其它超参

- **预测长度**：`config.horizons = {"short": 90, "long": 365}`，二者自动各自独立训练；
- **输入窗口**：`config.input_len = 90`；
- **实验轮数 / epoch**：`config.num_runs`、`config.epochs`，也可命令行覆盖：
  ```bash
  python main.py --epochs 60 --runs 5 --data_dir ./
  ```

### 4. 查看结果

- 控制台 / `outputs/results_summary.txt`：三模型在短期、长期任务上的 `MSE / MAE (Mean±Std)`；
- `outputs/forecast_short_90d.png`、`outputs/forecast_long_365d.png`：
  Ground Truth 与三模型预测曲线对比（用于报告“结果与分析”章节截图）。

---

## 四、重要的学术诚实说明

- **短序列长程预测的固有限制**：数据总长约 1400 天，而长期任务单个样本需
  `input(90)+output(365)=455` 天。严格“非重叠时间段”切分无法同时为 train/test 留足长度，
  因此本项目采用**样本级按时间顺序切分**（先按时间生成全部滑窗，再 8:1:1 划分
  train/val/test，绝不打乱）。边界处相邻样本会有少量重叠，这是该数据长度下做 365 天
  预测的客观限制，撰写报告“讨论”章节时应予以说明。
- **缺失值**：真实数据存在缺失属正常现象，已用前向填充处理，不影响预测任务。
- **可复现性**：每轮固定 `random / numpy / torch` 种子。
- **报告撰写工具**：允许使用 ChatGPT / DeepSeek 等辅助撰写文字部分，请按课程要求注明；
  必要的参考文献不可或缺。

## 五、参考文献（示例，请按实际补充）

1. UCI Machine Learning Repository: Individual household electric power consumption Data Set.
2. Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation, 1997.
3. Vaswani et al., *Attention Is All You Need*, NeurIPS, 2017.
