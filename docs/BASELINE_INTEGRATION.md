# 外部基线接入与目录重构

本次按 [V1规范](../specs/V1-项目结构重构与四类外部基线接入.md) 实施。改动前快照为 GitHub `main` 的 `1fbada44c526599f37bdbfac3f4b3fd72291eb02`。旧论文指标、已归档源码包和数据集未改写。

## 模型与代码位置

| 模型 | 主干入口 | 专用模块 | 原生点预测结构 |
|---|---|---|---|
| HADT-NQ | `models/hadt_nq.py` | `layers/hadt_nq/` | 保留已有原生分位数路径 |
| PatchTST-NQ | `models/patchtst.py` | `layers/patchtst/` | 监督版channel-independent patch Transformer与flatten head |
| TimeMixer-NQ | `models/timemixer.py` | `layers/timemixer/` | 多尺度分解混合、多尺度点预测及融合 |
| DLinear-NQ | `models/dlinear.py` | `layers/dlinear/` | 移动平均分解、季节/趋势线性预测相加 |
| iTransformer-NQ | `models/itransformer.py` | `layers/itransformer/` | 变量token、跨变量注意力、原生预测投影 |

`models/linear_nq.py`、`patch_nq.py`、`dense_nq.py`、`fixed_nq.py` 是原论文内部对照，不代表上述外部模型。旧实验工厂、结构搜索分别迁到 `experiments/paper_controls.py` 和 `experiments/architecture_search.py`。

公共损失、指标、RevIN、输出头已分别迁到 `utils/losses.py`、`utils/metrics.py`、`utils/revin.py`、`utils/quantile_heads.py`。旧 `losses/`、`metrics/` 和 `models/` 中层文件不再提供导入入口，仓库内引用已一起迁移。

## 统一分位数适配

```text
原生完整点模型（含原始预测头、原生逆归一化）
  → 点预测 [B,H,C]
  → [B×C,H,1]
  → MedianAnchoredNonCrossingHead(d_model=1)
  → [B,H,C,7]
```

四个外部模型复用同一个头类，各自独立训练参数；不扩展原生预测层到H×7，不附加高维升维MLP，不冻结主干，也不额外加MSE辅助损失。区间模式端到端优化现有七分位数pinball，验证pinball最小的检查点用于测试。`q50` 由追加头学习，不强制等于原点输出。

HADT保持原先D维特征接头、撤销RevIN的顺序，所以“头类相同”不代表头输入信息量或量纲完全相同。新增模型称为 `*-NQ` 分位数适配，不声称这些是官方原版概率模型。点模式仅用于主干保真和独立点预测检查，使用MSE，不输出区间指标。

所有输出在训练集标准化尺度，评估与绘图再撤销全局scaler。输入没有未来真实值或新增时间标记。

## 官方来源和支持范围

来源清单 [BASELINE_SOURCES.json](../third_party/BASELINE_SOURCES.json) 固定了四个上游提交、原始文件哈希、定义提取和本地映射。许可全文位于 `third_party/licenses/`。

- PatchTST：<https://github.com/yuqinie98/PatchTST>，`204c21efe0b39603ad6e2ca640ef5896646ab1a9`。
- TimeMixer：<https://github.com/kwuking/TimeMixer>，`e24610583b36fdd8c76cc17a8df4e65759a5f460`。
- DLinear：<https://github.com/cure-lab/LTSF-Linear>，`0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6`。
- iTransformer：<https://github.com/thuml/iTransformer>，`c2426e68ca13f74aaec08045c5c724d8ad328124`。

保留实际预测所需的官方模块，导入改成独立命名空间，不需要Reformer等无关模型依赖。iTransformer的输入归一化除法由原地改为非原地，避免历史输入梯度失败；同权重前向逐位一致。

首版支持long-term forecasting、M→M；TimeMixer限定moving-average分解、avg/max下采样，尺度长度必须整除，未来时间特征关闭。不开放未验证的DFT、卷积下采样、分类、插补等分支。模型专有参数见 `utils/baseline_config.py`；未支持参数明确报错。

iTransformer、TimeMixer、PatchTST默认完整变量前向；DLinear共享头可按变量分块，individual模式禁止。所有外部模型默认全变量训练。旧HADT保留原有变量抽样策略；强制要求不安全的抽样/分块会报错。绘图对跨变量模型先完整推理再选择展示变量。

## 运行

使用已有环境，不需要新增核心训练依赖：

```bash
conda activate djjTF
python -m unittest discover -s tests
python main.py --mode smoke --model patchtst --config configs/baselines/patchtst.yaml
```

四个模型的正式单次运行命令形式相同，修改配置文件名即可：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python main.py \
  --mode single --config configs/baselines/patchtst.yaml \
  --dataset ETTh1 --pred_len 96 --seed 2024
```

将`patchtst`替换为`timemixer`、`dlinear`、`itransformer`。配置默认30轮上限、验证集早停、seed2024/H96/ETTh1；其他长度、数据集和种子需显式安排。CPU可将配置中的 `runtime.device` 设为`cpu`。

旧 `main.py --mode smoke` 仍走原HADT smoke。旧无`model.name`的YAML仍构建原Transformer HADT；显式的新HADT配置可使用 `model.params.encoder_kind: mlp`，不会自动读取并替换为选中MLP JSON。

有限集成验证入口：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 python scripts/run_baseline_integration.py \
  --run-name baseline_integration_v1_release
```

该命令按 `configs/baseline_comparison.yaml`，对四个外部模型各训练1轮，使用完整ETTh1训练/验证/测试划分。运行记录明确标记`integration_only`，不能当作充分训练的模型效果或论文主结果。查看`outputs/baseline_integration_v1_release/index.html`和`integration_summary.json`。

## 保存、缓存与旧实验

新增有名模型运行路径：

```text
outputs/<run_name>/<model>/<variant>/<dataset>/H<h>/seed<s>/
```

包括最佳权重、完整解析配置、原生主干配置、源码/数据身份、训练曲线、逐步指标、固定首中末窗口的原始单位预测、PNG/PDF及浏览页面。`complete.json`只在必要产物全部落盘并哈希后写入。缓存命中会校验产物；配置、数据、源码或产物不一致时拒绝覆盖，请使用新的`run_name`。

每次进程只运行一个相同目录的任务，不并发写同一实验。配置本身不承担正式调参预算公平性的证明；正式论文比较需另外预先确定对等预算。

旧检查点通过原HADT入口strict加载；新检查点验证模型名、输出模式、variant和通道元数据。历史冻结脚本中的来源校验仍会拒绝把重构后的源码冒充原来源。重跑原始冻结研究可检出快照；不要强行重签旧记录。

汇总新旧预测浏览页：

```bash
python scripts/plot_run_forecasts.py --extra-root outputs/baseline_comparison \
  --gallery 预测结果浏览.html
```

新实验目录自带`figures/predictions/index.html`；新模型/variant不会再仅显示为项目名。

## 验证证据

- 原有38项测试迁移前后通过。
- 扩展后49项测试通过，覆盖统一标量头、梯度、主干点投影保留、通道策略、73变量输入、原始单位绘图、point完整链路、检查点元信息、缓存冲突和产物篡改拒绝。
- 原始Transformer和已选MLP的真实HADT检查点strict加载，参数键、输出和辅助量与迁移前快照逐位一致；可通过 `python scripts/verify_legacy_migration.py` 重做核验。
- 四个上游模型共8种配置与固定提交的官方代码在隔离子进程中同权重比较，最大绝对误差0；包括DLinear独立头、PatchTST分解及独立头、iTransformer无归一化及attention输出、TimeMixer变量耦合与max下采样。
- 数值参考脚本：`scripts/verify_baseline_parity.py --upstream <目录>`。该目录下需有四个仓库，名称分别为`patchtst/timemixer/dlinear/itransformer`，checkout到来源清单提交。参考子进程对未使用的可选Reformer导入提供会在调用时报错的占位符，实际FullAttention路径不调用它；报告明确记录此处理。
- GPU3单轮集成验证与检查报告保存在`outputs/baseline_refactor_validation/`。可审阅摘要在 `docs/validation/baseline_refactor_validation.json`。

本次未开展正式多种子外部基准矩阵，未更新论文排名或指标。
