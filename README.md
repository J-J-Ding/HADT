# HADT-NQ: Independent Research Codebase

2026-11-05：GPU 3 架构搜索与缺项实验已新增，可持续更新论文。实验协议、运行状态、恢复与核验方式见 [docs/ARCHITECTURE_STUDY.md](docs/ARCHITECTURE_STUDY.md)。实时完成情况以 `HADT_NQ_ICCRD2027_LaTeX/results/architecture_status.json` 为准。

预测曲线：在服务器项目目录打开 `预测结果浏览.html`，按数据集/长度/模型/种子浏览；各次实验的PNG与PDF位于对应运行目录的 `figures/predictions/`。

**Horizon-Adaptive Deformable Transformer with Non-Crossing Quantile Regression**

This is a standalone PyTorch research project. It does not depend on DBFI-Net or its training framework.

## 1. Main design

- Input: multivariate history `x: [B, L, C]`
- Output: seven conditional quantiles `q: [B, H, C, 7]`
- Quantiles: `[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]`
- Backbone: overlapping patch embedding + temporal Transformer encoder
- Decoder: horizon-conditioned deformable sparse attention
- Output head: median-anchored cumulative Softplus, structurally non-crossing
- Loss: multi-quantile pinball loss
- Data scaling: StandardScaler fitted on training rows only; model-side RevIN is reversible and has no affine parameters.

## 2. Project structure

```text
HADT_NQ/
├── configs/
│   └── default.yaml
├── data_provider/
│   ├── csv_dataset.py
│   ├── data_factory.py
│   └── scaler.py
├── engine/
│   ├── checkpoint.py
│   ├── evaluator.py
│   └── trainer.py
├── experiments/
│   └── runner.py
├── losses/
│   └── quantile_loss.py
├── metrics/
│   └── interval_metrics.py
├── models/
│   ├── hadt_nq.py
│   ├── horizon_deformable.py
│   ├── noncrossing_head.py
│   ├── patch_embedding.py
│   ├── revin.py
│   └── transformer_encoder.py
├── scripts/
│   ├── run_experiment.sh
│   └── run_smoke_test.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── utils/
│   ├── config.py
│   ├── logger.py
│   ├── seed.py
│   └── smoke_test.py
├── main.py
├── PROJECT_SPEC.md
└── requirements.txt
```

## 3. Environment

Recommended:
- Python 3.10 or 3.11
- PyTorch 2.2+
- CUDA-enabled PyTorch if training on NVIDIA GPU

Install:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 4. Dataset placement

Place CSV files in `./dataset/` by default:

```text
dataset/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── electricity.csv
├── traffic.csv
└── weather.csv
```

The file names can be changed in `configs/default.yaml`.

Each CSV must contain:
- one timestamp column (default name: `date`);
- numeric feature columns;
- no missing/non-finite values in selected numeric columns.

The timestamp column is excluded from `C`. All remaining numeric columns are used in multivariate (`M`) mode. The loader reports the detected feature names and `C`. If a dataset has a different timestamp column, update `data.time_column`.

## 5. Data split and window semantics

Default split follows the study's existing protocol: ETT datasets use chronological 60% / 20% / 20%; Electricity, Traffic, and Weather use 70% / 10% / 20%. The scaler is fitted only on the training portion.

For a target start index `t`:
- input is `data[t-L:t]`;
- target is `data[t:t+H]`.

Validation targets begin at the validation boundary; their input context may use the immediately preceding training history. Test targets begin at the test boundary; their input context may use preceding observed history. No target window crosses its own split boundary.

If a different split protocol is required for a particular benchmark, change the ratios in the YAML and record that choice in the paper.

## 6. Hyperparameters

The default main-experiment configuration is:
- `L=336`
- `H in [96, 192, 336, 720]`
- `P=16`, `S=8`
- `D=128`, `heads=8`, `encoder_layers=3`, `d_ff=256`
- dropout `0.1`
- sampling points `K=auto`
- maximum offset `1.5` Patch tokens
- seven quantiles as listed above

`C` is detected from the CSV. `N`, `K`, and `head_dim` are derived, not manually duplicated.

For `L=336`, `P=16`, `S=8`:
`N=floor((336-16)/8)+1=41`, and the default auto rule gives `K=8`.

## 7. Run commands

### Smoke test (no dataset needed)

```bash
python main.py --mode smoke
```

This checks:
- output shapes;
- strict quantile ordering;
- sampling positions within valid bounds;
- gradient flow to the offset generator;
- multiple channel counts and prediction horizons.

### One experiment

```bash
python main.py --mode single --dataset ETTh1 --pred_len 96 --seed 2024

# PowerShell uses the same command syntax.
```

### Full experiment matrix

```bash
python main.py --mode experiment
```

The full matrix is `datasets × horizons × seeds`. It can be expensive. Run one experiment first, inspect the logs, then launch the matrix.

### Run tests

```bash
python -m unittest discover -s tests -v
```

## 8. Memory controls

- Training samples a random subset of variables only when `C > channel_sample_size`. The same channel indices are used for `x` and `y`.
- Evaluation visits all variables in channel chunks.
- The deformable decoder processes the horizon dimension in chunks.
- Optional activation checkpointing recomputes each decoder chunk during backward to reduce saved activation memory.

These are memory-management choices. They do not change the mathematical output shape or the shared channel-independent model parameters.

## 9. Output artifacts

Each run writes to:

```text
outputs/HADT_NQ/<dataset>/H<pred_len>/seed<seed>/
```

Typical outputs:
- `train.log`
- `best_checkpoint.pt`
- `resolved_config.json`
- `metrics.json`
- `horizon_metrics.csv`

The experiment runner also writes `outputs/HADT_NQ/summary.csv`.

## 10. Important methodological notes

1. Non-crossing is guaranteed by the output parameterization; calibrated coverage is **not** guaranteed.
2. PICP is marginal elementwise coverage pooled over forecast windows and variables. It is not simultaneous joint coverage of the full multivariate vector.
3. The model is channel-independent with shared weights. It does not explicitly model cross-variable dependence.
4. Use validation data for model selection and hyperparameter decisions. Do not tune on test metrics.
5. Report multiple seeds and mean ± standard deviation.
6. The default values are a reasoned starting configuration, not a claim of globally optimal hyperparameters.


## 11. Inspect a dataset before training

```bash
python scripts/inspect_dataset.py --csv dataset/ETTh1.csv --time_column date
```

This reports row count, detected variable count, feature names, and finite-value status.

## 12. ETTh1：GPU 3 训练、预测图与模块贡献分析

当前数据目录为 `dataset/ETT-small/ETTh1.csv`。`configs/etth1_gpu3.yaml` 保留原模型结构和 60/20/20 时间划分，使用 L=336、H=96、seed=2024、batch=32、最多 30 轮、验证集早停 7 轮。为使用显存，关闭梯度重计算。与原 batch=4 的优化轨迹会有所不同。

```bash
conda activate djjTF
pip install -r requirements.txt
# 仅暴露物理 GPU 3；程序内 cuda:0 即这张卡。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 \
  python scripts/run_etth1_analysis.py --config configs/etth1_gpu3.yaml

# 只训练完整模型（仍包含三种推理阶段干预）
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 \
  python scripts/run_etth1_analysis.py --variants full

# 从已经完成、且配置一致的实验继续；不会恢复未完成的训练。
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 \
  python scripts/run_etth1_analysis.py --reuse-existing

# 普通单次实验同样自动保存可视化
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  python main.py --config configs/etth1_gpu3.yaml --mode single --dataset ETTh1 --pred_len 96 --seed 2024
```

训练、验证、测试均有 tqdm 进度条，显示批次、耗时和预计剩余时间；训练显示累计 pinball 损失。各轮统计持久保存为 `training_history.csv`。验证集损失最小时保存 `best_checkpoint.pt`，含模型、优化器、调度器、AMP scaler 状态、训练集标准化参数、特征名、配置和最佳轮次；测试前重新加载该模型。它可用于恢复模型推理，未提供完整随机状态和训练恢复入口。

分析入口依次训练 `full`、`no_offsets`、`no_encoder`、`no_revin`，每次使用相同数据、seed 和训练预算，分别根据各自验证集选择最佳模型。输出目录为：

```text
outputs/ETTh1_analysis/
  component_comparison.csv         # 重新训练消融与推理干预分别标注
  component_comparison.png/.pdf
  full/ETTh1/H96/seed2024/
    best_checkpoint.pt
    resolved_config.json          # 数据划分、标准化参数和全部配置
    training_history.csv
    metrics.json
    horizon_metrics.csv
    train.log
    figures/
      forecast_00_HUFL.png/.pdf    # 每个变量单独一张，固定首/中/尾窗口
      forecast_06_OT.png/.pdf
      forecast_samples.npz        # 原始量纲历史、标签、7 个分位数和样本编号
      training_history.png/.pdf
      calibration.png/.pdf
      intermediate_steps.png/.pdf # RevIN、Patch、Encoder、偏移、采样位置、注意力
      intermediate_steps.npz      # 三个固定测试窗口的中间输出
    interventions/                # 最佳完整模型关闭单个模块后的全测试集指标
  no_offsets/ETTh1/H96/seed2024/   # 下同，各自独立训练和保存
  no_encoder/ETTh1/H96/seed2024/
  no_revin/ETTh1/H96/seed2024/
```

如何判断贡献：先看完整模型与**重新训练消融**的测试 pinball（越低越好）、覆盖率与区间宽度，不能只看覆盖率。`inference_intervention` 是对已经训练好的模型移除模块，衡量依赖性，可能引入分布变化，不等于重新训练消融。热图揭示模型行为，不能单独证明模块有效。非交叉输出结构保证分位数排序，但不保证覆盖率校准；本轮是单 seed 探索，不能据此声称统计显著性。ETT 当前 60/20/20 划分与常见固定月份基准协议不同，比较其他论文时须使用相同划分。

## 13. Traffic 编码器 0–3 层对比

```bash
conda activate /home/dingjunjie/anaconda3/envs/djjTF
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 \
  MPLCONFIGDIR=/tmp/hadt_nq_mpl python scripts/run_traffic_depth_comparison.py
```

配置为 `configs/traffic_gpu3.yaml`：L=336、H=96、seed=2024、batch=32、最多30轮、验证早停7轮。Traffic 沿用项目的70/10/20时间划分，训练每批随机取16个变量；验证和测试遍历全部862个变量。0层跳过整个编码器，包括末尾LayerNorm，与ETTh1实验定义一致。

结果保存在 `outputs/Traffic_depth_comparison/`：`comparison.csv`、`RESULTS.md`、`depth_comparison.png/.pdf`、`learning_curves.png/.pdf`、`forecast_OT_depths.png/.pdf`，以及和ETTh1对照的 `cross_dataset_comparison.csv`、`cross_dataset_relative_loss.png/.pdf`。各层子目录含最佳检查点、训练历史、完整测试指标和预测图。

高维绘图可通过 `visualization.max_channels` 限制图示变量数，按原始变量索引均匀选取，独立于预测效果；NPZ保存 `channel_indices`。此选项只限制绘图，不影响训练或指标统计。不设置时保持原有全部变量出图行为。

已完成且配置、数据指纹一致的运行会被复用；修改CSV后拒绝混用旧结果。单seed跨数据集比较可以观察层数结论的一致性，不能代替多seed稳定性实验。
