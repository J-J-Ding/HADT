# 实测范围与完成状态

2026-11-05：历史稿件的两小时范围限制不再作为本轮队列限制。新增架构搜索与七数据集×四长度×三种子评估的进度以`architecture_status.json`为准，新增已完成记录见`architecture_records.json`。下文66条记录说明原稿，不表示本轮尚在运行的扩展已经完成。

本稿按用户两小时交付约束限定研究范围，已完成66条种子实验记录及所有正文中列出的表格。完整实验矩阵见 evidence_runs.csv，正文统计见 aggregate_metrics.csv 与 sampling_sensitivity_aggregates.csv。

此前模板建议的七数据集、四预测长度全矩阵未作为本稿范围；本稿结果使用ETTh1、ETTh2的重复研究，以及明确标注的Traffic单次案例。所有实验图表由实际完成记录生成；不以未完成运行填充空缺。

已核验：所有66条记录的指标来源、最优检查点、训练前缀scaler、数据SHA、固定图示样本真值、量化维度。均值和SD由完整种子集合生成；不适用于独立头的proper interval metrics未进入比较。
