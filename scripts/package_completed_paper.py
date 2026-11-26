"""Package the measured manuscript, evidence and a compact reproducibility source archive."""
import hashlib,json,re,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
import importlib.metadata as metadata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scripts.run_paper_experiments import code_digest

P=Path('HADT_NQ_ICCRD2027_LaTeX')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    records=json.loads((P/'results/evidence_records.json').read_text())
    sensitivity=json.loads((P/'results/sampling_sensitivity_records.json').read_text())
    validation=json.loads((P/'results/artifact_validation.json').read_text())
    assert validation['status']=='passed' and validation['completed_records']==len(records)+len(sensitivity)
    protocol=json.loads(Path('outputs/paper_study/protocol.json').read_text())
    assert code_digest()==protocol['source_hash'],'Frozen model/training source changed'
    completed=len(records)+len(sensitivity)
    prior=sum('prior_run' in r.get('source_kind','') for r in records)
    bib=(P/'references.bib').read_text()
    keys=re.findall(r'@\w+\s*\{\s*([^,]+)',bib)
    assert len(keys)==len(set(keys))==15
    sources=(P/'main.tex').read_text()+''.join(p.read_text() for p in (P/'sections').glob('*.tex'))
    cited=set()
    for chunk in re.findall(r'\\cite\w*\{([^}]+)\}',sources):cited.update(chunk.split(','))
    assert cited<=set(keys),f'Missing bibliography: {cited-set(keys)}'
    audit=json.loads((P/'results/reference_audit.json').read_text())
    assert audit['reference_count']==15
    versions={name:metadata.version(name) for name in ['torch','numpy','pandas','matplotlib','PyYAML','tqdm']}
    versions.update(python=sys.version.split()[0],GPU='NVIDIA GeForce RTX 5090',physical_gpu_index=3,
                    CUDA_runtime='12.8',environment_name='djjTF',timing_note='GPU was shared; no isolated hardware speed claim')
    (P/'results/software_environment.json').write_text(json.dumps(versions,indent=2))
    (P/'results/citation_check.json').write_text(json.dumps(dict(status='passed',bib_entries=len(keys),
            cited_keys=sorted(cited),unresolved_keys=[]),indent=2))
    readme=f'''# HADT-NQ / ICCRD 2027 完整实证论文

已使用提供的 ACM `sigconf,anonymous,review` 模板完成英文匿名论文。入口为 **main.pdf**，源码为 **main.tex** 与 sections/。正文包含引言、相关工作、方法及有序性说明、实验设计、实测结果、讨论、局限和结论。

## 实验范围与证据

- 共整理 **{completed} 条已完成的种子实验记录**：主研究 {len(records)} 条、K 敏感性 {len(sensitivity)} 条。其中 {prior} 条为配置/数据匹配后核验复用的既有结果，历史代码指纹明确标为未知；其余为本轮实际训练。
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
'''
    (P/'README_CN.md').write_text(readme,encoding='utf-8')
    (P/'results/RESULTS_REQUIRED.md').write_text(f'''# 实测范围与完成状态

本稿按用户两小时交付约束限定研究范围，已完成{completed}条种子实验记录及所有正文中列出的表格。完整实验矩阵见 evidence_runs.csv，正文统计见 aggregate_metrics.csv 与 sampling_sensitivity_aggregates.csv。

此前模板建议的七数据集、四预测长度全矩阵未作为本稿范围；本稿结果使用ETTh1、ETTh2的重复研究，以及明确标注的Traffic单次案例。所有实验图表由实际完成记录生成；不以未完成运行填充空缺。

已核验：所有{completed}条记录的指标来源、最优检查点、训练前缀scaler、数据SHA、固定图示样本真值、量化维度。均值和SD由完整种子集合生成；不适用于独立头的proper interval metrics未进入比较。
''',encoding='utf-8')
    checklist='''# 评审稿交付核验

- 使用用户提供的ACM sigconf匿名评审模板，英文正文与匿名PDF元数据。
- 论文中的数值由完整实测记录生成，主对照/长度扩展/消融/采样敏感性与单次案例分别标注。
- 数据划分、训练前缀标准化、验证选优与全部变量测试均已核验。
- 15个文献条目真实可追溯，全部引用key存在；DOI与官方出处见reference_audit。
- 使用嵌入字体，未使用Type3图表字体；图例、表头、公式与引用经渲染检查。
- 作者/机构仍为匿名评审状态；论文尚未提交至会议系统。

官方格式与评审来源：https://www.iccrd.org/sub.html 。公开页面要求英文、使用模板及双盲评审，未公开数值页数上限。当前文稿页数以main.pdf为准；提交时以会务门户要求为准。
'''
    (P/'SUBMISSION_CHECKLIST.md').write_text(checklist,encoding='utf-8')
    (P/'figures/README.md').write_text('''# 正文图示

paper_architecture：模型示意，无实测数据。
paper_calibration：ETT完整三种子组的覆盖率与一个样本SD。
paper_forecasts：预先固定的首个测试窗口，原始量纲真实值、预测中位数和90%区间。
paper_sampling：ETTh1 seed2025首个测试窗口、OT变量、head0的实际offset与attention。
paper_depth_pilot：既有ETTh1/Traffic seed2024深度案例，不含虚构误差条。

全部正文图PDF使用嵌入字体。数据来源见results/figure_sources.json；PNG仅用于预览。原fig1/fig2/fig3为模板遗留示意，未用于新版正文。
''',encoding='utf-8')
    (P/'supplementary').mkdir(exist_ok=True)
    reproduce='''# Reproduction and evidence

The study uses PyTorch and the package versions in results/software_environment.json. Place the same CSV versions at dataset/ETT-small/ETTh1.csv, dataset/ETT-small/ETTh2.csv and dataset/traffic/traffic.csv; verify the SHA256 values in results/datasets.json. Dataset files and checkpoints are not included in this compact source archive.

From the project root, with the required CUDA-enabled Python environment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python scripts/run_paper_experiments.py
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 python scripts/run_paper_sampling_sensitivity.py
OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/hadt_paper_mpl python -m unittest discover -s tests -v
MPLCONFIGDIR=/tmp/hadt_paper_mpl python scripts/export_paper_evidence.py
python scripts/validate_paper_artifacts.py
```

GPU3 was used in the recorded study. For a machine exposing only GPU0, add PAPER_EXPECTED_GPU=0 and set CUDA_VISIBLE_DEVICES=0. This changes the local device guard, not the model.

The experiment runner falls back to fresh training when old pilot outputs are absent. Completed cache entries require matching configuration/data/implementation signatures. Two imported historical records lack original source hashes and are labeled accordingly. Canonical provenance signs the pre-fit configuration; generated scaler/feature-column fields are retained separately in resolved_config.json. Raw original markers are preserved.

Software unit tests contain synthetic scoring/shape examples only; these are never loaded as paper forecast results. Tables use measured completed runs, not synthetic scores. Three-seed groups require exactly2024/2025/2026; sample SD usesddof1. Independent-head interval metrics are excluded from proper-score comparisons.

The primary run queue additionally provides single-seed Traffic controls. Figures of the earlier depth pilot require the saved CSVs in results/depth_pilot_etth1.csv and depth_pilot_traffic.csv; local full-project scripts can reproduce those pilots. The source archive includes the related scripts. Exporter paths refer to the full project layout, while the LaTeX directory is independently compilable.

A fresh run can take much longer than the original delivery window, depending on hardware, shared GPU load and cache reuse. The recorded deadline is a scheduling limit; no model group is padded with unfinished seeds.
'''
    (P/'supplementary/REPRODUCE.md').write_text(reproduce)
    root_doc=Path('/tmp/paper_repro_readme.md');root_doc.write_text(reproduce)
    script_names=['run_paper_experiments.py','run_paper_sampling_sensitivity.py','export_paper_evidence.py',
                  'validate_paper_artifacts.py','draw_paper_architecture.py','run_depth_comparison.py',
                  'run_traffic_depth_comparison.py','plot_cross_dataset_depths.py','build_paper.sh',
                  'inspect_dataset.py','run_smoke_test.py']
    code_files=[]
    for d in ['models','data_provider','engine','layers','utils','experiments','configs','tests']:
        code_files.extend(p for p in Path(d).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml'))
    code_files.extend(Path('scripts')/n for n in script_names)
    code_files.extend([Path('requirements.txt'),Path('main.py')])
    code_zip=P/'supplementary/reproducibility_source.zip'
    with zipfile.ZipFile(code_zip,'w',zipfile.ZIP_DEFLATED) as z:
        for p in code_files:z.write(p,p.as_posix())
        z.write(root_doc,'README_REPRODUCIBILITY.md')
    # Include only active paper sources/evidence; legacy template ZIP and unused diagrams remain local.
    paper_files=[P/n for n in ['main.tex','main.pdf','references.bib','acmart.cls','ACM-Reference-Format.bst',
                              'README_CN.md','SUBMISSION_CHECKLIST.md','compile_windows.bat']]
    paper_files.extend((P/'sections').glob('*.tex'))
    paper_files.extend(p for p in (P/'figures').glob('paper_*') if p.suffix in ('.pdf','.png'))
    paper_files.append(P/'figures/README.md')
    paper_files.extend(p for p in (P/'results').rglob('*') if p.is_file() and p.name!='delivery_receipt.json' and p.suffix in ('.tex','.csv','.json','.md','.npz'))
    paper_files.extend([code_zip,P/'supplementary/REPRODUCE.md'])
    manifest=dict(completed_utc=datetime.now(timezone.utc).isoformat(),title='HADT-NQ',
                  benchmark_records=completed,newly_trained_records=completed-prior,historical_imports=prior,
                  references=15,anonymous=True,measured_scope='ETTh1/ETTh2 repeated study; Traffic single-seed case',
                  frozen_model_training_source_sha256=code_digest(),
                  files={str(p.relative_to(P)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paper_files))})
    (P/'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    target=Path('HADT_NQ_ICCRD2027_Completed.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paper_files+[P/'PACKAGE_MANIFEST.json'])):z.write(p,p.as_posix())
    (P/'results/delivery_receipt.json').write_text(json.dumps(dict(package=str(target),package_sha256=sha(target),
        paper_sha256=sha(P/'main.pdf'),benchmark_records=completed,references=15),indent=2))
    print('Packaged',target,'bytes',target.stat().st_size)
    print('Verified records',completed,'new',completed-prior,'historical',prior,'references',len(keys),'cited',len(cited))

if __name__=='__main__':main()
