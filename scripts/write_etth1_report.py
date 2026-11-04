"""Summarize saved experiments without selecting or training on test results."""
from pathlib import Path
import json
import hashlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch


def main():
    root = Path('outputs/ETTh1_analysis')
    labels = dict(full='完整模型', no_offsets='关闭可变形偏移', no_encoder='去掉 Transformer', no_revin='去掉 RevIN')
    lines = ['# ETTh1 实验结果（2026-10-30）', '',
             '环境：djjTF；物理 GPU 3（RTX 5090）；历史 336 小时预测未来 96 小时；7 变量；seed=2024。',
             '按时间顺序 60%/20%/20% 划分，训练集拟合 StandardScaler；每个模型独立按验证损失选优。最多 30 轮，早停耐心 7，batch=32。', '',
             '## 重新训练的模块消融', '',
             '| 模型 | 最佳轮次 / 实际轮次 | 验证 pinball | 测试 pinball ↓ | 90%覆盖率 | 90%归一化区间宽度 |',
             '|---|---:|---:|---:|---:|---:|']
    metrics = {}
    for variant in labels:
        run = root / variant / 'ETTh1/H96/seed2024'
        values = json.loads((run / 'metrics.json').read_text())
        checkpoint = torch.load(run / 'best_checkpoint.pt', map_location='cpu', weights_only=False)
        history = pd.read_csv(run / 'training_history.csv')
        metrics[variant] = values
        lines.append(f"| {labels[variant]} | {checkpoint['epoch']} / {len(history)} | {values['best_val_pinball']:.6f} | {values['QuantileLoss_standardized']:.6f} | {values['PICP90']:.2%} | {values['PINAW90']:.6f} |")
    base = metrics['full']['QuantileLoss_standardized']
    lines += ['', '相对完整模型的测试 pinball 变化（正数表示移除后变差）：', '']
    for variant in ['no_offsets','no_encoder','no_revin']:
        delta = (metrics[variant]['QuantileLoss_standardized'] / base - 1)*100
        lines.append(f'- {labels[variant]}：{delta:+.2f}%。')
    lines += ['', '## 已训练模型上的推理干预', '',
              '下表只关闭完整模型的模块，不重新训练，反映模型对模块的依赖，不能替代上面的重训消融。', '',
              '| 关闭模块 | 测试 pinball | 90%覆盖率 |', '|---|---:|---:|']
    for variant in ['no_offsets','no_encoder','no_revin']:
        m = json.loads((root / f'full/ETTh1/H96/seed2024/interventions/{variant}/metrics.json').read_text())
        lines.append(f"| {labels[variant]} | {m['QuantileLoss_standardized']:.6f} | {m['PICP90']:.2%} |")
    lines += ['', '## 预测与中间过程', '',
              '![OT 首/中/尾三个固定测试窗口](full/ETTh1/H96/seed2024/figures/forecast_06_OT.png)', '',
              '红线为预测中位数、黑线为真实值、蓝色为 50%/80%/90% 区间。全部变量均有独立图片和 PDF，NPZ 保留原始量纲数据。样本按测试集首/中/尾固定选取。',
              'OT 中部窗口存在预测偏高，尾部升温阶段预测偏弱；不能因为区间大致覆盖就判定趋势预测准确。', '',
              '![关键步骤](full/ETTh1/H96/seed2024/figures/intermediate_steps.png)', '',
              '依次展示 RevIN、Patch 嵌入、Transformer 输出、偏移热图、采样位置和注意力权重。偏移随预测步变化说明模型使用了偏移机制，其收益应结合重训指标判断。', '',
              '![训练曲线](full/ETTh1/H96/seed2024/figures/training_history.png)', '',
              '![校准曲线](full/ETTh1/H96/seed2024/figures/calibration.png)', '',
              '![组件比较](component_comparison.png)', '',
              '## 结果边界与复现', '',
              '本轮为单数据集、H=96、单 seed 的探索性结果。模块效果不能据此声称统计显著性。区间指标汇总了全部测试窗口、预测步和变量，属于边际覆盖率；不代表联合覆盖率。',
              '当前划分与 ETT 常见固定月份协议不同，不能直接与使用另一协议的论文指标比较。覆盖率须结合宽度和 pinball 判断；非交叉输出保证排序，不保证 90% 区间达到 90% 覆盖率。', '',
              '复现命令：', '', '```bash', 'conda activate djjTF',
              'CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 \\',
              '  python scripts/run_etth1_analysis.py --config configs/etth1_gpu3.yaml',
              'python scripts/write_etth1_report.py', '```', '',
              '环境版本见 `environment.json`；每个子目录的 `resolved_config.json` 保存数据边界、变量和标准化参数，`best_checkpoint.pt` 保存验证集最优参数。', '',
              '数据 SHA256：`' + hashlib.sha256(Path('dataset/ETT-small/ETTh1.csv').read_bytes()).hexdigest() + '`。', '',
              '验证：8 项单元/集成测试通过，GPU 3 冒烟测试通过，pip check 无依赖冲突；独立静态代码审查未发现阻塞问题。']
    (root / 'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(root / 'RESULTS.md')


if __name__ == '__main__':
    main()
