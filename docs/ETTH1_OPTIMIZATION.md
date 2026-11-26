# ETTh1 结构、历史长度与参数优化

本轮按照用户要求，使用四张GPU探索更稳健的HADT配置，关注过拟合、整体分位数质量和局部变化，不改写历史实验结果。设计见 [V2规范](../specs/V2-ETTh1四预测长度结构与参数搜索.md)。

## 实验规则

1. ETTh1原60/20/20时间划分，scaler仅拟合训练前缀。
2. 历史长度96/336/512先做独立预实验，比较同一reference结构在H96/192/336/720上的验证指标，再固定一个全局历史长度。
3. 固定历史长度后，18个预先声明的候选分别在四个预测长度筛选，每组seed2024、最多6轮。
4. 综合验证分数前三，加reference保留，去重后各做四个预测长度×三个种子2024/2025/2026的最多30轮确认，按验证pinball早停选模。
5. 冻结精度最优、细节平衡最优后才评估测试。不同H分别训练模型；比较中的历史长度和结构超参数相同，不是从同一个H720模型截取其他长度。

局部指标：中位数一阶差分MAE；真实变化超过训练前缀绝对一阶差分95%分位数时的中位数MAE；总变差比作为诊断。目标变化中的不可预测噪声不应被强行复现，不能只追求曲线更抖动。

综合分数在每个H相对reference归一化，然后等权平均：`0.7*pinball_ratio + 0.2*delta_mae_ratio + 0.1*jump_mae_ratio`。细节配置还要求平均验证pinball在精度最优的2%以内，各H覆盖率不比reference低超过2个百分点。所有条件在测试前声明。

## 新增代码

- `models/hadt_detail.py`：保留原HADT，提供本轮实验变体。
- `layers/hadt_nq/local_detail.py`：轻量局部/空洞depthwise卷积编码块、近期加密参考网格。
- `utils/detail_objective.py`：可选训练期差分辅助损失；默认权重0时等于原pinball。验证始终使用proper pinball。
- `utils/forecast_detail_metrics.py`：训练前缀事件阈值、变化误差及区间诊断。
- `experiments/detail_search.py`：完整候选表、配置生成和验证排名。

中位数残差使用`Linear(L,H)`直接读取RevIN归一化历史，经dropout和有界门控形成同一个平移量，加到所有分位数，保留顺序。linear和seasonal候选结构相同，分别零初始化和24小时季节复制初始化；后者不是固定季节朴素模型，权重仍端到端学习。局部卷积/残差是候选，不能在验证前就宣称有效或最终采用。

## 运行与查看

```bash
conda activate djjTF
python scripts/run_etth1_detail_search.py --gpus 0 1 2 3
# 搜索完成、selection.json冻结之后，补充按H的验证部署选择：
python scripts/freeze_etth1_horizon_choices.py
python scripts/evaluate_etth1_detail_search.py --gpus 0 1 2 3
python scripts/report_etth1_detail_search.py
```

默认目录：`outputs/etth1_detail_search/`。

- `progress.json`：当前阶段与完成数量；`workers/`保存各GPU日志和任务。
- `protocol.json`：启动前冻结的候选、预算、数据/实现指纹。
- `history_selection.json`、`shortlist.json`、`selection.json`：仅验证数据驱动的选择。
- 每个运行保存`resolved_config.json`、最佳权重、训练曲线、验证/训练诊断及完整产物校验。
- 最终`report.html`、`index.html`、`test_aggregates.csv`、`learning_curves.png/pdf`及`local_detail_H*.png/pdf`由后两步生成。

不要在搜索过程中编辑已冻结的模型、公共工具或搜索控制器；身份冲突会拒绝复用旧记录，不应重新签名冒充同一次运行。历史配置和历史模型仍保留。

最终推荐配置写入`configs/optimized_etth1.yaml`，另保留accuracy/detail两份配置。复现所声明训练目标（尤其可选差分项）使用：

```bash
python scripts/retrain_etth1_selected.py --config configs/optimized_etth1.yaml \
  --root outputs/optimized_etth1_retrain --gpus 0 1 2 3
```

本轮是有限候选和预算内的优化，不是全局最优证明；短预算初筛可能漏掉慢收敛候选，多次使用验证集带来选择乐观性，三个种子不是统计显著性保证。只在测试后报告实际效果，不因测试不理想修改选择。

确认阶段增加了按预测长度的验证部署选择（在首次测试前冻结），原定全局推荐仍保留。`horizon_selection.json`记录该补充及各H配置；`configs/optimized_etth1_H96.yaml`等四份配置保持同一历史长度，但允许结构参数不同，不把拼接成绩当作统一结构结果。

## 已完成结果

已完成12组历史窗口预实验、72组初筛、48组三种子确认训练，以及入选4个结构的48组完整测试。历史长度固定为336；55项测试通过，旧Transformer和MLP检查点回归通过，新reference与旧已选MLP的初始化和前向逐位一致。来源、数据与保存产物哈希均已核验。

全局精度和细节规则均选择`linear_skip`：两层MLP、D64/FFN128、patch16/stride8、8头×8采样点、最大偏移1.5、dropout0.2、weight decay0.001、学习率0.0003、batch32；额外的历史到中位数修正支路为无bias的`Linear(336,H)`，零初始化，支路dropout0.1，有界门控初始值0.1。该修正对全部分位数加同一个量，不破坏顺序。训练上限30轮、验证早停7，参数变化与结构变化作为一套配置比较，不把收益全部归因于单个模块。

| H | 原已选MLP Pinball | 全局linear_skip Pinball | 按H验证精度优选 | 该配置测试Pinball | 变化MAE改善 |
|---|---:|---:|---|---:|---:|
| 96 | 0.132541 | 0.129663 | seasonal_skip | 0.129758 ± 0.000743 | 8.71% |
| 192 | 0.144811 | 0.141383 | linear_skip | 0.141383 ± 0.001341 | 11.10% |
| 336 | 0.157559 | 0.153758 | linear_skip | 0.153758 ± 0.001694 | 14.17% |
| 720 | 0.186469 | 0.186827 | local_fine | 0.183754 ± 0.000774 | 7.26% |

均值与样本标准差来自2024/2025/2026三个种子。表中按H配置不同，不是一个结构的统一成绩。H96的seasonal_skip与linear_skip结构相同，区别为支路线性权重采用24小时季节初始化；H720的local_fine为patch8/stride4、两层局部卷积残差编码器，无中位数修正支路，其余主要超参数与正则化设置相同。

局部卷积为depthwise kernel3＋特征MLP，带初始0.1的LayerScale。最大历史长度仍为336，不能将“小patch”误解为输入窗口缩短。

### 配置入口

- `configs/optimized_etth1.yaml`：全局统一的linear_skip配置，覆盖四个H。
- `configs/optimized_etth1_H96_accuracy.yaml`、`H192_accuracy.yaml`、`H336_accuracy.yaml`、`H720_accuracy.yaml`：各H的验证精度优选，均使用L336。
- `configs/optimized_etth1_H96.yaml`等无accuracy后缀的四份文件：各H的验证细节平衡优选；H720为seasonal_skip，它的变化误差改善更大，但测试Pinball略差于reference，不应当作H720精度最优。
- `outputs/etth1_detail_search/report.html`：完整结果与分析。
- `outputs/etth1_detail_search/index.html`：各模型、各H、各种子的完整预测区间图。
- `docs/validation/etth1_optimization_*`：便于版本管理的选择规则、统计及校验摘要。

### 实际局限

全局linear_skip在H720的测试Pinball略差于reference（约0.19%），虽然变化误差明显降低，不能称为四个长度全面领先。按H精度优选的H720局部卷积Pinball改善约1.46%，但大变化时刻MAE没有改善（约恶化0.12%）。区间仍存在覆盖不足：按H精度优选PICP90约为86.56%、85.72%、85.45%、82.73%，均未达到90%。因此本轮改善了部分预测细节与精度，没有证明消除过拟合、解决极端事件或完成校准。

配置选择在测试前冻结，未因上述测试表现重选参数。按H细节平衡规则与精度规则同时保留，用户可按部署目标取舍。短预算筛选、有限候选及重复验证使用的局限仍然存在。
