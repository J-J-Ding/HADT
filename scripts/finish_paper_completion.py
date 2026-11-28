"""Audit all four complete datasets and publish tables; refuses partial groups."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from utils.provenance import sha256,atomic_json
from scripts.prepare_revised_paper import table,pm
ROOT=Path('outputs/paper_completion_20261126')
EXT=Path('outputs/paper_seven_datasets_20261126')
P=Path('HADT_NQ_ICCRD2027_LaTeX')
DATASETS=('ETTh1','ETTh2','ETTm1','ETTm2','Traffic','Weather','Electricity')
records=[]
for phase,count in [('ablation',36),('baseline',48)]:
    rows=json.loads((ROOT/f'{phase}_records.json').read_text());assert len(rows)==count
    records.extend(rows)
extension=json.loads((EXT/'records.json').read_text());assert len(extension)==360
records.extend(extension)
inventory=[]
for r in records:
    run=Path(r['source_dir'])
    for f,digest in r['artifacts'].items():assert sha256(run/f)==digest
    hist=pd.read_csv(run/'training_history.csv')
    expected_cap=10 if r['job'].get('dataset') in ('Traffic','Weather','Electricity') else 30
    assert r['job']['epochs']==expected_cap and len(hist)<=expected_cap
    assert abs(r['best_val']-hist.val_pinball.min())<1e-9
    assert r['best_epoch']==int(hist.loc[hist.val_pinball.idxmin(),'epoch'])
    if r['job']['kind'] in ('baseline','full'):
        resolved=json.loads((run/'resolved_config.json').read_text())
        cfg=resolved['config'];job=r['job']
        inventory.append(dict(dataset=job.get('dataset','ETTh1'),model=job['name'],horizon=job['horizon'],seed=job['seed'],
            n_parameters=resolved['n_parameters'],train_batch_size=cfg['training']['batch_size'],
            evaluation_batch_size=cfg['evaluation']['batch_size'],
            train_channel_policy=cfg['training'].get('channel_policy','auto')))
def aggregate(rows):
    f=pd.DataFrame(rows);assert len(f)==3
    assert set(f.seed)=={2024,2025,2026}
    out={}
    for key in f.select_dtypes(include='number'):
        if key=='seed':continue
        out[key+'_mean']=float(f[key].mean());out[key+'_std']=float(f[key].std(ddof=1))
    return out

def crossing_mean(row):
    value=row['CrossingRate_mean']*100
    if value==0:return '$0$'
    mantissa,exponent=f'{value:.2e}'.split('e')
    return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
full=[]
for h in (96,192,336,720):
    for seed in (2024,2025,2026):
        run=Path(f'outputs/etth1_detail_search/confirm/L336/linear_skip/H{h}/seed{seed}')
        r=json.loads((run/'test_complete.json').read_text())
        for f,d in r['artifact_sha256'].items():assert sha256(run/f)==d
        full.append(dict(dataset='ETTh1',name='hadt',horizon=h,seed=seed,**{**r['test'],**r['test_detail']}))
        resolved=json.loads((run/'resolved_config.json').read_text());cfg=resolved['config']
        inventory.append(dict(dataset='ETTh1',model='hadt',horizon=h,seed=seed,
            n_parameters=resolved['trainable_parameters'],train_batch_size=cfg['training']['batch_size'],
            evaluation_batch_size=cfg['evaluation']['batch_size'],train_channel_policy=cfg['training'].get('channel_policy','auto')))
primary=[dict(dataset=r['job'].get('dataset','ETTh1'),name=r['job']['name'],horizon=r['job']['horizon'],
             seed=r['job']['seed'],**{**r['metrics'],**r['details']})
         for r in records if r['job']['kind'] in ('baseline','full')]
allrows=primary+full;assert len(allrows)==420
labels={'dlinear':'DLinear-NQ','patchtst':'PatchTST-NQ','timemixer':'TimeMixer-NQ','itransformer':'iTransformer-NQ','hadt':'HDA-NQ'}
aggregates={}
for dataset in DATASETS:
    for name in labels:
        for h in (96,192,336,720):
            aggregates[dataset,name,h]=aggregate([r for r in allrows if r['dataset']==dataset and r['name']==name and r['horizon']==h])
rows=[]
for dataset in DATASETS:
    for i,h in enumerate((96,192,336,720)):
        means=[aggregates[dataset,n,h]['QuantileLoss_standardized_mean'] for n in labels]
        best=min(means);cells=[dataset if i==0 else '',str(h)]
        for name in labels:
            g=aggregates[dataset,name,h];s=pm(g,'QuantileLoss_standardized')
            if g['QuantileLoss_standardized_mean']==best:s=r'\boldmath '+s
            cells.append(s)
        rows.append(' & '.join(cells)+r' \\')
    if dataset!=DATASETS[-1]:rows.append(r'\midrule')
table('revised_external.tex','Seven-dataset common-head comparison: mean $\\pm$ sample SD of test pinball over three seeds. '
      'Every model uses $L=336$; each dataset and $H$ has separately trained models. Structures and learning rates are frozen from ETTh1 validation. '
      'Bold marks the lowest mean within each row.',
      'tab:external',['Dataset','$H$',*labels.values()],rows,'llrrrrr',True)
intervalrows=[];interval_means={}
for dataset in DATASETS:
    for i,(name,label) in enumerate(labels.items()):
        values=[r for r in allrows if r['dataset']==dataset and r['name']==name]
        f=pd.DataFrame(values).groupby('seed')[['PICP90','PINAW90','NormalizedWinkler90']].mean()
        g={}
        for key in f:g[key+'_mean']=float(f[key].mean());g[key+'_std']=float(f[key].std(ddof=1))
        interval_means[dataset,name]=g
        intervalrows.append(' & '.join([dataset if i==0 else '',label,pm(g,'PICP90',2,100),pm(g,'PINAW90',4),pm(g,'NormalizedWinkler90',4)])+r' \\')
    if dataset!=DATASETS[-1]:intervalrows.append(r'\midrule')
table('supplement_external_intervals.tex','Complete interval statistics, averaging four horizons within each seed before computing mean $\\pm$ sample SD across three seeds. '
      'All models use ordered outputs. Coverage is assessed relative to 90\\%; width must be read together with coverage.',
      'tab:full-intervals',['Dataset','Model',r'PICP90 (\%)','PINAW90','Norm. IS90'],intervalrows,'llrrr',False)
compact=[]
for dataset in DATASETS:
    cells=[dataset]
    for name in labels:
        g=interval_means[dataset,name]
        cells.append('$'+f'{100*g["PICP90_mean"]:.2f}'+r'\,/\,'+f'{g["NormalizedWinkler90_mean"]:.4f}'+'$')
    compact.append(' & '.join(cells)+r' \\')
table('revised_external_intervals.tex','Coverage and interval score: each cell is PICP90 (\\%) / normalized IS90, '
      'averaged equally over four horizons and three seeds. Coverage is assessed against 90\\%; lower IS is preferable. '
      'Widths and sample standard deviations appear in Supplementary Table S3.',
      'tab:external-intervals',['Dataset',*labels.values()],compact,'lrrrrr',True)
rows=[];ablationstats=[]
for name,label in [('full','Full HDA-NQ'),('fixed','Fixed offsets'),('shared','Shared embedding'),('dense','Dense attention'),
                    ('no_skip','No correction'),('fixed_no_skip','Fixed + no correction'),('independent','Independent head')]:
    cells=[label]
    for h in (96,720):
        group=[dict(seed=r['job']['seed'],**{**r['metrics'],**r['details']}) for r in records
               if r['job']['kind']=='ablation' and r['job']['name']==name and r['job']['horizon']==h]
        if name=='full':group=[r for r in full if r['horizon']==h]
        g=aggregate(group)
        cells.extend([pm(g,'QuantileLoss_standardized'),crossing_mean(g)])
        ablationstats.append(dict(variant=name,horizon=h,**g))
    rows.append(' & '.join(cells)+r' \\')
table('supplement_full_ablation.tex','Complete retrained component comparisons on ETTh1 at both studied horizons. Pinball is mean $\\pm$ sample SD over three seeds; crossing is the mean percentage. '
      'all settings except the named component match HDA-NQ. Crossing is the percentage of forecasts with any adjacent reversal. '
      'Independent outputs are not assigned interval scores.',
      'tab:full-ablation',['Variant',r'\multicolumn{2}{c}{$H=96$}',r'\multicolumn{2}{c}{$H=720$}'],
      [r' & Pinball & Crossing (\%) & Pinball & Crossing (\%) \\',r'\midrule',*rows],'lrrrr',False)

ab={(r['variant'],r['horizon']):r for r in ablationstats}
variant_labels=[('full','Full HDA-NQ'),('fixed','Fixed offsets'),('shared','Shared embedding'),('dense','Dense attention'),
                ('no_skip','No correction'),('fixed_no_skip','Fixed + no correction'),('independent','Independent head')]
focused=[' & '.join([label,pm(ab[name,96],'QuantileLoss_standardized'),crossing_mean(ab[name,96])])+r' \\' for name,label in variant_labels]
table('revised_final_ablation.tex','Focused component analysis of 96-step long-term forecasting on ETTh1. Pinball is mean $\\pm$ sample SD over three seeds; crossing is the mean percentage. '
      'The complete $H=96/720$ results are reported in Supplementary Table S1.',
      'tab:final-ablation',['Variant','Pinball',r'Crossing (\%)'],focused,'lrr',False)
details=[' & '.join([label,pm(ab[name,96],'delta_mae',4),pm(ab[name,720],'delta_mae',4),
                    pm(ab[name,96],'jump_mae',4),pm(ab[name,720],'jump_mae',4)])+r' \\' for name,label in variant_labels]
table('supplement_detail_ablation.tex','Complete local-variation diagnostics for the same component groups, mean $\\pm$ sample SD. '
      'Event thresholds are the training-prefix 95th percentiles of absolute changes.',
      'tab:detail-ablation',['Variant',r'$\Delta$MAE 96',r'$\Delta$MAE 720','Event MAE 96','Event MAE 720'],details,'lrrrr',False)
for name in ('supplement_full_ablation.tex','supplement_detail_ablation.tex'):
    path=P/'results'/name
    path.write_text(path.read_text().replace(r'\begin{table}[t]',r'\begin{table}[htbp]'))

# Interpret measured means without pooling losses from different target distributions.
wins={name:sum(aggregates[d,'hadt',h]['QuantileLoss_standardized_mean']<aggregates[d,name,h]['QuantileLoss_standardized_mean']
               for d in DATASETS for h in (96,192,336,720)) for name in labels if name!='hadt'}
best=sum(aggregates[d,'hadt',h]['QuantileLoss_standardized_mean']==min(aggregates[d,n,h]['QuantileLoss_standardized_mean'] for n in labels)
         for d in DATASETS for h in (96,192,336,720))
analysis='HDA-NQ achieves the lowest mean pinball in '+str(best)+' of the 28 dataset--horizon settings. '
analysis+='It has lower mean loss than '+', '.join(labels[n]+' in '+str(wins[n])+'/28 settings' for n in wins)+'. '
analysis+='These counts describe within-setting comparisons; they are not statistical significance tests or an average of losses across datasets. '
analysis+='Table~\\ref{tab:external-intervals} reports coverage and interval scores alongside fit, so narrower intervals are not interpreted as gains in isolation.\n'
(P/'results/completion_comparison_analysis.tex').write_text(analysis)
fixed_gain=[100*(1-ab['full',h]['QuantileLoss_standardized_mean']/ab['fixed',h]['QuantileLoss_standardized_mean']) for h in (96,720)]
shared_gain=[100*(1-ab['full',h]['QuantileLoss_standardized_mean']/ab['shared',h]['QuantileLoss_standardized_mean']) for h in (96,720)]
delta_change=[100*(ab['no_skip',h]['delta_mae_mean']/ab['full',h]['delta_mae_mean']-1) for h in (96,720)]
abtext=(f'Learned positions reduce mean pinball relative to fixed references by {fixed_gain[0]:.2f}\\% at $H=96$ and {fixed_gain[1]:.2f}\\% at $H=720$. '
        f'The corresponding reductions relative to a shared horizon embedding are {shared_gain[0]:.2f}\\% and {shared_gain[1]:.2f}\\%. '
        f'Removing the correction raises first-difference error by {delta_change[0]:.2f}\\% and {delta_change[1]:.2f}\\%, respectively, '
        'but slightly improves 720-step pinball. Thus, preserving local variation does not guarantee a lower quantile loss. '
        'Independent outputs exhibit rare nonzero crossings and a slightly lower 720-step loss; the ordered head supplies a structural guarantee rather than a uniform accuracy advantage.\n')
(P/'results/supplement_ablation_analysis.tex').write_text(abtext)
focused_text=(f'At $H=96$, learned positions reduce mean pinball relative to fixed references by {fixed_gain[0]:.2f}\\%, '
              f'and future-step embeddings reduce it relative to a shared embedding by {shared_gain[0]:.2f}\\%. '
              f'Removing the historical correction raises first-difference error by {delta_change[0]:.2f}\\%. '
              'These focused results describe the 96-step long-term forecasting setting. The complete two-horizon results and component tradeoffs are retained in the supplement; uniform gains across horizons are not inferred.\n')
(P/'results/completion_ablation_analysis.tex').write_text(focused_text)
pd.DataFrame([dict(dataset=d,model=n,horizon=h,**g) for (d,n,h),g in aggregates.items()]).to_csv(P/'results/completion_baseline_aggregates.csv',index=False)
pd.DataFrame(ablationstats).to_csv(P/'results/completion_ablation_aggregates.csv',index=False)
pd.DataFrame(allrows).to_csv(P/'results/completion_baseline_runs.csv',index=False)
assert len(inventory)==420
pd.DataFrame(inventory).to_csv(P/'results/completion_model_inventory.csv',index=False)
for source,target in [(ROOT/'protocol.json','completion_etth1_protocol.json'),(ROOT/'selection.json','completion_selection.json'),
                      (ROOT/'model_checks.json','completion_model_checks.json'),(EXT/'protocol_v2.json','completion_transfer_protocol.json'),
                      (EXT/'epoch_cap_amendment.json','completion_epoch_cap_amendment.json')]:
    atomic_json(P/'results'/target,json.loads(source.read_text()))
atomic_json(P/'results/completion_audit.json',dict(status='passed',datasets=list(DATASETS),primary_test_runs=420,primary_three_seed_groups=140,
    new_screen_runs=12,new_ablation_runs=36,new_baseline_runs=336,new_final_model_runs=72,new_training_jobs=456,
    reused_final_runs=12,group_policy='Exactly 3 seeds per published group',test_used_for_selection=False,
    selection_scope='Model and hyperparameter selection; not qualitative display selection.',
    display_policy='Main component table focuses on H96 after inspection; complete H96/H720 results retained in supplement.',
    frozen_protocol_sha256=sha256(ROOT/'protocol.json'),baseline_selection_sha256=sha256(ROOT/'selection.json'),
    transfer_protocol_sha256=sha256(EXT/'protocol_v2.json'),source_artifacts_verified=True,validation_best_epochs_verified=True))
print('Seven datasets, all horizons and seeds audited; complete tables published.')
(P/'results/completion_scope.tex').write_text('The completed comparison includes all seven benchmark datasets and all four prediction lengths: 420 test runs in 28 complete five-model, three-seed settings. Table~\\ref{tab:data} summarizes the datasets.\n')
table('completion_dataset_table.tex','Datasets in the complete comparison. Splits are chronological.',
      'tab:data',['Dataset','Rows','Variables','Split'],
      [r'ETTh1 / ETTh2 & 17,420 & 7 & 60/20/20 \\',
       r'ETTm1 / ETTm2 & 69,680 & 7 & 60/20/20 \\',
       r'Traffic & 17,544 & 862 & 70/10/20 \\',
       r'Weather & 52,696 & 21 & 70/10/20 \\',
       r'Electricity & 26,304 & 321 & 70/10/20 \\'],'lrrl',False)

# Refresh quantitative summaries after the full matrix finishes, preserving Introduction.
main=P/'main.tex'
text=main.read_text()
old_intro=text.split(r'\section{Introduction}',1)[1].split(r'\input{sections/related_work}',1)[0]
import re
text=re.sub(r'Across 12 completed dataset--horizon settings on four ETT datasets, HDA-NQ attains the lowest mean pinball loss in 11 settings',
            f'Across 28 dataset--horizon settings on seven datasets, HDA-NQ attains the lowest mean pinball loss in {best} settings',text)
text=text.replace('The completed comparisons on four ETT datasets favor the proposed model in 11 of 12 dataset--horizon settings',
                  f'The complete comparisons on seven datasets favor the proposed model in {best} of 28 dataset--horizon settings')
text=text.replace(' Conclusions about Traffic, Weather, and Electricity await their complete experimental matrices.','')
assert text.split(r'\section{Introduction}',1)[1].split(r'\input{sections/related_work}',1)[0]==old_intro
main.write_text(text)
exp=P/'sections/experiments.tex'
text=exp.read_text().replace('The completed comparisons reported here cover four related ETT datasets. The broader benchmark protocol also includes Traffic, Weather, and Electricity; their incomplete comparisons are excluded from this revision.',
                            'The completed comparison covers seven public datasets, including four related ETT series.')
exp.write_text(text)
supp=P/'supplement.tex'
text=supp.read_text().replace('Only dataset--horizon settings with all five models and three seeds complete are included, matching the main comparison. The horizon sets are listed in the main interval table; these are averaged within each training seed before computing the three-seed mean and sample standard deviation.',
                             'All seven datasets, five models, and four prediction lengths are included. Horizons are averaged within seed before computing the three-seed mean and sample standard deviation.')
supp.write_text(text)

from scripts.style_paper_tables import format_tables
format_tables(P/'results/completion_baseline_runs.csv',P/'results/completion_baseline_aggregates.csv')
