# GPU 3 架构搜索与论文同步

本轮使用 `/home/dingjunjie/anaconda3/envs/djjTF/bin/python`，仅暴露物理 GPU 3。GPU 0–2 不参与。

## 实验顺序

1. ETTh1、H96、seed2024，预先固定的12个候选各筛选8轮。比较Transformer、MLP、膨胀CNN、CNN+Transformer、直接输出等组合，另有D64与学习率3e-4的联合变体。
2. 验证损失前三名加原模型，各用seed2024/2025/2026、最多30轮复核。按**平均验证集pinball**冻结最佳候选。此时尚未用测试集决定配置。
3. 冻结后评估选中模型、原模型及Linear-NQ。七数据集×四预测长度(96/192/336/720)×三种子；选中模型与对照相同时去重。先做ETTh1/ETTh2，再做新增数据集。完整矩阵可能耗时较长，待运行项不作为已有结果。

每次运行保留配置、数据/代码/检查点SHA256、最佳验证检查点、训练曲线和全部测试窗口/变量的指标。示例预测固定为测试集首、中、末窗口，不能用图示窗口替代全测试指标。

首轮短暂运行归档于`outputs/architecture_study_pilot_20261105/`，其目录中保存的源路径是归档前路径；不参与新实验统计。新版本改用与标准Conv1d在CPU双精度数值和梯度上对照一致的unfold+linear卷积实现，绕开当前硬件/确定性cuDNN对膨胀卷积的慢内核。独立代码审查发现的配置冻结、缓存与论文证据绑定缺口已修复并有回归测试。

## 查看实时状态

```bash
cat outputs/architecture_study/current_job.json
cat HADT_NQ_ICCRD2027_LaTeX/results/architecture_status.json
tail -f outputs/architecture_study/worker.log
```

主进程PID保存在`outputs/architecture_study/worker.pid`，单独会话运行，终端断开不会自动停止。`failure.json`表示发生错误，恢复时会清除旧错误。`progress.json`记录测试矩阵进度；筛选与复核状态见论文目录的`architecture_status.json`。启动锁阻止重复工作进程。

## 恢复队列

仅在原进程已退出后执行，完成且溯源匹配的单次实验会复用；中断的单次训练从头重新训练，而非恢复优化器随机状态。

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 \
  MPLCONFIGDIR=/tmp/hadt_architecture_mpl \
  /home/dingjunjie/anaconda3/envs/djjTF/bin/python -u scripts/run_architecture_study.py
```

代码、配置或数据改变后会拒绝混用旧目录。需要新的实验命名空间，而不是删除原始证据。

## 论文与结果

每次完成实验后自动调用`export_architecture_study.py`，更新：

冻结选型、首次测试完成及每新增完整三种子组时，会自动调用独立核验脚本；核验失败则停止队列，不发布未经核验的新PDF。

- `results/architecture_section.tex`：搜索协议、候选表、完成的三种子测试表和已测结论。
- `results/architecture_records.json`、`architecture_aggregates.csv`、`architecture_status.json`：原始记录、完整组均值/样本标准差、进度。
- `figures/architecture_*.pdf/.png`：筛选、验证学习曲线、选中结构与固定测试窗口预测图。
- `main.pdf`：通过本地Tectonic重新编译。
- `configs/selected_architecture.json`：冻结后写入选中候选及其训练配置。

原论文默认模型的历史结果和结构图保留，选中变体另行标明。局部验证搜索无法证明全局最优；测试迁移不改善时也应如实保留。直接输出模型胜出不能据此宣称形变注意力有效或组件具有新颖性。

独立核验当前已完成记录（CPU）：

```bash
CUDA_VISIBLE_DEVICES='' /home/dingjunjie/anaconda3/envs/djjTF/bin/python \
  scripts/validate_architecture_study.py
```

核验包括数据与检查点哈希、训练前缀scaler、划分边界、最佳验证轮次、固定图示真值、排序、完整种子统计、选型冻结指纹；不会重新推理所有完整测试集。训练时不应同时手动运行导出器，因为它和工作进程会写相同的论文文件。
