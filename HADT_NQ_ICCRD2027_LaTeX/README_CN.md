# HADT-NQ / ICCRD 2027 完整实证论文

2026-11-05 扩展说明：下文的66条记录及原交付ZIP/manifest描述历史稿件。本轮另增验证集架构搜索及七数据集、四预测长度的队列；实时完成状态见`results/architecture_status.json`，新增证据见`architecture_records.json`，正文见`architecture_section.tex`。只有已完成的三种子组进入新增汇总。工作区`main.pdf`会随已完成实验持续重编译。协议、恢复和独立核验方式见项目根目录`docs/ARCHITECTURE_STUDY.md`。旧的`package_completed_paper.py`锁定历史版本，不能用于打包本轮扩展。

已使用提供的 ACM `sigconf,anonymous,review` 模板完成英文匿名论文。入口为 **main.pdf**，源码为 **main.tex** 与 sections/。正文包含引言、相关工作、方法及有序性说明、实验设计、实测结果、讨论、局限和结论。

## 实验范围与证据

- 共整理 **66 条已完成的种子实验记录**：主研究 60 条、K 敏感性 6 条。其中 2 条为配置/数据匹配后核验复用的既有结果，历史代码指纹明确标为未知；其余为本轮实际训练。
- 主对照：ETTh1、ETTh2，H=96，HADT-NQ / Linear-NQ / Patch-NQ / Dense-NQ / Fixed-NQ，各3个种子。
- 长度扩展：两ETT数据集，H=192，HADT-NQ / Linear-NQ / Dense-NQ，各3个种子。
- 关键消融：ETTh1 H=96，共享查询、独立输出头、关闭RevIN，各3个种子。
- 采样点数：ETTh1 H=96，K=4/8/12，各3个种子；K=8复用主对照，不据测试结果重选默认参数。
- Traffic：HADT-NQ / Linear-NQ / Dense-NQ，H=96、seed=2024；测试全部862个变量，明确为单种子案例。
- 既有0–3层深度案例保持单种子定位，不冒充多种子稳定性结论。

三种子集合为2024/2025/2026。均值与样本标准差(ddof=1)只由完整三种子组产生，误差条不是置信区间。独立头的区间指标不适合作为proper interval comparisons，聚合中留空，仅比较损失与交叉率；原始诊断数值仍可追溯。

## 关键文件

- results/aggregate_metrics.csv：主研究19个完整三种子组的未舍入统计量。
- results/evidence_runs.csv、evidence_records.json：逐运行结果与源路径。
- results/sampling_sensitivity_aggregates.csv：K敏感性统计。
- results/reference_audit.md/.json：15篇真实文献的来源、DOI或官方页面与核验信息。
- results/artifact_validation.json：检查点选优、训练集标准化、数据指纹和图片真值对应核验。
- results/provenance_audit.json：逐运行签名及历史来源限制。
- results/figure_sources.json：图的数据来源、固定样本、变量和注意力头。
- results/figure_data/：校准图数据和预测/采样图的实测NPZ数组，可直接检查绘图原值。
- results/model_inventory.csv、software_environment.json：参数数量和实际软件环境。
- supplementary/reproducibility_source.zip：精简代码、配置、测试和复现入口，未包含大型数据与检查点。

本工作区的原始训练日志、完整检查点、逐步指标与NPZ位于 outputs/paper_study/ 和 outputs/paper_sensitivity/。数据文件版本由 results/datasets.json 的SHA256锁定。未完成的可选运行不进入论文结果。

## 阅读结论

结果并非所有对照都优于简单模型：HADT-NQ在ETT中改善了相对Dense-NQ与Fixed-NQ的部分指标，Linear-NQ/Patch-NQ在一些比较中更强；Traffic的Dense-NQ损失更低。论文保留这些结果，并区分排序保证、经验覆盖率和预测精度。ETT采用60/20/20时间划分，与常见固定月份协议不同，不混用其他论文的公开分数。

## 编译

在项目根目录可运行：

```bash
TECTONIC_OFFLINE=1 ./scripts/build_paper.sh
```

本地便携Tectonic已验证。也可在此目录使用标准TeX环境运行 `latexmk -pdf main.tex`，或依次pdflatex、bibtex、pdflatex两遍。保持原始acmart.cls与ACM-Reference-Format.bst。

复现实验与证据生成见 supplementary/REPRODUCE.md。当前保留匿名作者与机构字段，未填写DOI、ISBN或录用信息。文献核验与软件检查不等同于会议录用或外部同行评审。
