"""Publish only complete five-model, three-seed result rows from a fixed snapshot."""
from pathlib import Path
import json
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.provenance import sha256, atomic_json

P = Path('HADT_NQ_ICCRD2027_LaTeX/results')
BASE = Path('outputs/paper_completion_20261126')
EXT = Path('outputs/paper_seven_datasets_20261126')
LABELS = {'dlinear':'DLinear-NQ','patchtst':'PatchTST-NQ','timemixer':'TimeMixer-NQ',
          'itransformer':'iTransformer-NQ','hadt':'HDA-NQ'}
DATASETS = ('ETTh1','ETTh2','ETTm1','ETTm2','Traffic','Weather','Electricity')


def table(name, caption, label, headers, rows, alignment, wide=False):
    env = 'table*' if wide else 'table'
    (P/name).write_text('\n'.join([rf'\begin{{{env}}}[t]',r'\centering\small',
        r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\begin{tabular}{@{}'+alignment+'@{}}',r'\toprule',
        ' & '.join(headers)+r' \\',r'\midrule',*rows,r'\bottomrule',
        r'\end{tabular}',rf'\end{{{env}}}'])+'\n')


def aggregate(rows):
    assert len(rows)==3 and {r['seed'] for r in rows}=={2024,2025,2026}
    f=pd.DataFrame(rows)
    return {k+suffix:float(f[k].mean() if suffix=='_mean' else f[k].std(ddof=1))
            for k in f.select_dtypes(include='number') if k not in ('seed','horizon')
            for suffix in ('_mean','_std')}


def pm(g, key, digits=5, scale=1):
    return '$'+f'{g[key+"_mean"]*scale:.{digits}f}'+r'\pm'+f'{g[key+"_std"]*scale:.{digits}f}'+'$'


def load_record(path, reused=False):
    r=json.loads(path.read_text());run=path.parent
    for name,digest in r['artifact_sha256' if reused else 'artifacts'].items():
        assert sha256(run/name)==digest,(run,name)
    if reused:
        return dict(dataset='ETTh1',name='hadt',horizon=r['horizon'],seed=r['seed'],
                    **{**r['test'],**r['test_detail']})
    h=pd.read_csv(run/'training_history.csv')
    assert abs(r['best_val']-h.val_pinball.min())<1e-9
    assert r['best_epoch']==int(h.loc[h.val_pinball.idxmin(),'epoch'])
    j=r['job']
    return dict(dataset=j.get('dataset','ETTh1'),name=j['name'],horizon=j['horizon'],
                seed=j['seed'],**{**r['metrics'],**r['details']})


def main():
    paths=sorted((BASE/'baseline').glob('*/H*/seed*/complete.json'))
    paths+=sorted(EXT.glob('*/*/H*/seed*/complete.json'))
    raw=[load_record(p) for p in paths]
    full=[load_record(p,True) for p in sorted(Path('outputs/etth1_detail_search/confirm/L336/linear_skip').glob('H*/seed*/test_complete.json'))]
    raw+=full
    settings=[(d,h) for d in DATASETS for h in (96,192,336,720)
              if all(len([r for r in raw if (r['dataset'],r['horizon'],r['name'])==(d,h,n)])==3 for n in LABELS)]
    included=[r for r in raw if (r['dataset'],r['horizon']) in settings]
    assert len(included)==len(settings)*15
    groups={(d,h,n):aggregate([r for r in included if (r['dataset'],r['horizon'],r['name'])==(d,h,n)])
            for d,h in settings for n in LABELS}
    datasets=[d for d in DATASETS if any(x==d for x,h in settings)]
    rows=[]
    for d,h in settings:
        best=min(groups[d,h,n]['QuantileLoss_standardized_mean'] for n in LABELS)
        cells=[d,str(h)]
        for n in LABELS:
            g=groups[d,h,n];value=pm(g,'QuantileLoss_standardized')
            cells.append((r'\boldmath ' if g['QuantileLoss_standardized_mean']==best else '')+value)
        rows.append(' & '.join(cells)+r' \\')
    table('revised_external.tex','Completed ETT comparisons: test pinball mean $\\pm$ sample SD over three seeds. '
          'Every displayed row contains all five models and three seeds, with $L=336$. Lower is better; bold marks the lowest mean. '
          'The displayed horizons are determined by completed runs, not by their ranking.',
          'tab:external',['Dataset','$H$',*LABELS.values()],rows,'llrrrrr',True)
    intervalrows=[];expanded=[]
    for d in datasets:
        horizons=[h for x,h in settings if x==d]
        cells=[d,','.join(map(str,horizons))]
        for n,label in LABELS.items():
            f=pd.DataFrame([r for r in included if r['dataset']==d and r['name']==n])
            f=f.groupby('seed')[['PICP90','PINAW90','NormalizedWinkler90']].mean()
            g={k+s:float(f[k].mean() if s=='_mean' else f[k].std(ddof=1)) for k in f for s in ('_mean','_std')}
            cells.append('$'+f'{100*g["PICP90_mean"]:.2f}'+r'\,/\,'+f'{g["NormalizedWinkler90_mean"]:.4f}'+'$')
            expanded.append(' & '.join([d,label,pm(g,'PICP90',2,100),pm(g,'PINAW90',4),pm(g,'NormalizedWinkler90',4)])+r' \\')
        intervalrows.append(' & '.join(cells)+r' \\')
    table('revised_external_intervals.tex','Completed-setting interval evaluation: PICP90 (\\%) / normalized IS90. '
          'Only the listed horizons enter each dataset average, equally weighted within each seed. '
          'Coverage is assessed against 90\\%; lower interval score is preferable. Supplementary Table S3 gives widths and variability.',
          'tab:external-intervals',['Dataset','$H$',*LABELS.values()],intervalrows,'llrrrrr',True)
    table('supplement_external_intervals.tex','Complete interval statistics for the same completed settings as the main comparison. '
          'Listed horizons are averaged within seed before computing mean $\\pm$ sample SD over three seeds.',
          'tab:full-intervals',['Dataset','Model',r'PICP90 (\%)','PINAW90','Norm. IS90'],expanded,'llrrr')
    wins={n:sum(groups[d,h,'hadt']['QuantileLoss_standardized_mean']<groups[d,h,n]['QuantileLoss_standardized_mean'] for d,h in settings) for n in LABELS if n!='hadt'}
    best=sum(groups[d,h,'hadt']['QuantileLoss_standardized_mean']==min(groups[d,h,n]['QuantileLoss_standardized_mean'] for n in LABELS) for d,h in settings)
    text=f'HDA-NQ achieves the lowest mean pinball in {best} of the {len(settings)} completed dataset--horizon settings. '
    text+='It has lower mean loss than '+', '.join(LABELS[n]+f' in {wins[n]}/{len(settings)} settings' for n in wins)+'. '
    text+='These counts describe within-setting means, not statistical significance. Completion is a scheduling criterion rather than performance-based selection; no conclusion about the remaining settings is drawn.\n'
    (P/'completion_comparison_analysis.tex').write_text(text)
    scope=(f'This revision reports {len(included)} completed test runs across {len(datasets)} ETT datasets and {len(settings)} dataset--horizon settings. '
           'A comparison row is included only when all five models have finished all three seeds. '
           'The broader seven-dataset study introduced above is still in progress; the present tables make no claim about its unreported settings. '
           'Table~\\ref{tab:data} describes the datasets represented by the completed comparisons. ')
    (P/'completion_scope.tex').write_text(scope+'\n')
    data=[r'ETTh1 / ETTh2 & 17,420 & 7 & 60/20/20 \\',r'ETTm1 / ETTm2 & 69,680 & 7 & 60/20/20 \\']
    table('completion_dataset_table.tex','ETT datasets represented in the completed comparisons. Splits are chronological.',
          'tab:data',['Dataset','Rows','Variables','Split'],data,'lrrl')
    # Recompute the complete component study independently of extension progress.
    ablrecords=[load_record(p) for p in sorted((BASE/'ablation').glob('*/H*/seed*/complete.json'))]
    assert len(ablrecords)==36
    variants={'full':'Full HDA-NQ','fixed':'Fixed offsets','shared':'Shared embedding','dense':'Dense attention',
              'no_skip':'No correction','fixed_no_skip':'Fixed + no correction','independent':'Independent head'}
    ab={}
    for n in variants:
        for h in (96,720):
            group=[r for r in (full if n=='full' else ablrecords) if r['horizon']==h and (n=='full' or r['name']==n)]
            ab[n,h]=aggregate(group)
    def cross(g):
        value=g['CrossingRate_mean']*100
        if value==0:return '$0$'
        a,b=f'{value:.2e}'.split('e');return '$'+a+r'\times10^{'+str(int(b))+'}$'
    focused=[' & '.join([label,pm(ab[n,96],'QuantileLoss_standardized'),cross(ab[n,96])])+r' \\' for n,label in variants.items()]
    table('revised_final_ablation.tex','ETTh1 component study at $H=96$ (long-term forecasting). Mean $\\pm$ sample SD over three seeds; crossing is the mean percentage. Complete $H=96/720$ results appear in Supplementary Table S1.',
          'tab:final-ablation',['Variant','Pinball',r'Crossing (\%)'],focused,'lrr')
    complete=[];details=[]
    for n,label in variants.items():
        complete.append(' & '.join([label,*[v for h in (96,720) for v in (pm(ab[n,h],'QuantileLoss_standardized'),cross(ab[n,h]))]])+r' \\')
        details.append(' & '.join([label,pm(ab[n,96],'delta_mae',4),pm(ab[n,720],'delta_mae',4),pm(ab[n,96],'jump_mae',4),pm(ab[n,720],'jump_mae',4)])+r' \\')
    table('supplement_full_ablation.tex','Complete ETTh1 component study; pinball mean $\\pm$ sample SD and mean crossing percentage over three seeds.',
          'tab:full-ablation',['Variant','Pinball 96',r'Crossing 96 (\%)','Pinball 720',r'Crossing 720 (\%)'],complete,'lrrrr')
    table('supplement_detail_ablation.tex','Complete local-variation diagnostics, mean $\\pm$ sample SD over three seeds. Event thresholds use training-prefix changes.',
          'tab:detail-ablation',['Variant',r'$\Delta$MAE 96',r'$\Delta$MAE 720','Event MAE 96','Event MAE 720'],details,'lrrrr')
    pd.DataFrame(included).to_csv(P/'snapshot_baseline_runs.csv',index=False)
    pd.DataFrame([dict(dataset=d,horizon=h,model=n,**g) for (d,h,n),g in groups.items()]).to_csv(P/'snapshot_baseline_aggregates.csv',index=False)
    pd.DataFrame([dict(variant=n,horizon=h,**g) for (n,h),g in ab.items()]).to_csv(P/'completion_ablation_aggregates.csv',index=False)
    atomic_json(P/'snapshot_audit.json',dict(status='passed',created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        datasets=datasets,published_runs=len(included),complete_settings=settings,observed_complete_runs=len(raw),
        ablation_runs=36,source_artifacts_verified=True,validation_best_epochs_verified=True,
        selection_rule='Only full five-model three-seed rows; no performance filtering.',wins=wins,best=best,
        source_records={str(p):sha256(p) for p in paths}))
    from scripts.style_paper_tables import format_tables
    format_tables(P/'snapshot_baseline_runs.csv',P/'snapshot_baseline_aggregates.csv')
    print(json.dumps(dict(published_runs=len(included),complete_settings=settings,best=best,wins=wins),indent=2))


if __name__=='__main__':
    main()
