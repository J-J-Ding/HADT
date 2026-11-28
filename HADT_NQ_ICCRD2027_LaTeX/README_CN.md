# HADT-NQ 论文修订版

入口：`main.tex`；编译结果：`main.pdf`；完整消融补充材料为 `supplement.tex` / `supplement.pdf`。保留ACM sigconf匿名评审模板，按Introduction、Related Work、Method、Experiments、Conclusion组织。

## 模型与实验范围

- 正文模型HADT-NQ为最终统一linear_skip配置：两层D64残差MLP、预测步条件化可变形解码器、有序头和共同历史修正。
- 主比较覆盖ETTh1、ETTh2、ETTm1、ETTm2、Traffic、Weather、Electricity，历史336，预测96/192/336/720，5个模型，每组3种子，共420次完整测试。
- 外部baseline为DLinear-NQ、PatchTST-NQ、TimeMixer-NQ、iTransformer-NQ，保留已有点预测主干并接入共同分位数输出参数化。
- ETTh1另有6个消融变体、2个预测长度、3种子，共36组新增训练。
- 全部追加训练共456组：12个验证学习率预实验、336组外部baseline、72组新增数据集最终模型、36组消融；复用12组已冻结最终模型结果。
- 正文只描述最终模型。旧七数据集三层MLP研究和Transformer结果保留作历史记录，不混入本稿比较。
- ETT使用60/20/20、其他数据集使用70/10/20时间比例划分，与常见固定月份协议不同，不混用公开leaderboard分数。

## 来源与复现

- `results/completion_audit.json`：最终结构补实验及校验完成凭证。
- `results/completion_baseline_aggregates.csv`、`completion_ablation_aggregates.csv`：新实验完整统计。
- `results/revision_sources.json`：新版图表来源指纹。
- `results/reference_audit.md`：原15篇文献核验；`reference_additions_20261126.json`：新增5篇官方来源。
- `results/revision_validation.json`：最终编译、引用和完整性核验。
- 完整检查点和训练记录位于项目根目录 `outputs/`；交付包不包含大型检查点或数据集。

使用djjTF环境，依次执行：

```bash
python scripts/complete_paper_experiments.py
python scripts/extend_paper_datasets.py
python scripts/prepare_revised_paper.py
python scripts/finish_paper_completion.py
TECTONIC_OFFLINE=1 ./scripts/build_paper.sh
TECTONIC_OFFLINE=1 ./scripts/build_paper.sh HADT_NQ_ICCRD2027_LaTeX/supplement.tex
```

旧版自动导出器和交付ZIP保留作历史记录，不用于重建本修订稿。不要在补实验进行中改动冻结的模型、配置或控制器代码。

## 图件

正文使用 `figures/revised_architecture.pdf`、`revised_forecasts.pdf`、`revised_calibration.pdf` 和 `revised_attention.pdf`，对应框架、完整长度预测、完整测试覆盖率及固定窗口机制汇总。均有矢量与预览文件。详细讲解框架为 `paper_architecture_detailed.pdf`，附可编辑SVG。

原稿备份：项目 `outputs/paper_revision_20261126/before/`，保留用户此前修改。


## 脱离AI会话的自动执行

训练与监督均为本机Python进程，不调用OpenAI API。独立会话监督程序为 `scripts/paper_background_supervisor.py`，自动接续训练队列，并在实验齐全后执行结果核验、绘图、正文及补充材料编译、10页上限和引用检查、LaTeX交付包生成。

实时进度：项目 `outputs/paper_revision_20261126/progress.html`。
监督状态：`outputs/paper_revision_20261126/supervisor_status.json`。
完成状态：`outputs/paper_revision_20261126/finalization_progress.json`。

若本机程序异常，监督程序最多自动恢复三次，复用已完成且通过哈希检查的实验。未完成的单组训练会重新执行，不承诺从中断批次继续。遇到持续训练或版面错误，状态会标为需要处理并保留日志。服务器关机或所有监督进程被终止时，自动执行会停止。


2026-11-27训练预算调整：Traffic、Weather、Electricity的所有模型和预测长度统一上限10轮；ETT四数据集仍为30轮，保留验证早停及最佳权重。原冻结协议保存在protocol.json，新生效协议为protocol_v2.json，调整记录为epoch_cap_amendment.json。四组已开始的训练从保存的最佳检查点接续；旧检查点没有RNG状态，接续时使用记录的确定性重设种子，不宣称与未中断训练逐位相同。后续检查点保存随机状态。
