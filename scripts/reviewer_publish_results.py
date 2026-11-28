"""Publish complete V4 studies; keep detailed tables and plots in supplement."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def publish(paper,root):
    from utils.provenance import sha256,atomic_json
    paper=Path(paper);root=Path(root)
    if not (root/'records.json').exists():return False
    records=json.loads((root/'records.json').read_text());assert len(records)==51
    rows=[]
    for r in records:
        j=r['job']
        directory=j['id'] if j['phase']=='validation_pilot' else j['variant']
        run=root/j['phase']/directory/f"H{j['horizon']}"/f"seed{j['seed']}"
        for f,digest in r['artifacts'].items():assert sha256(run/f)==digest
        for f,digest in r.get('external_artifacts',{}).items():assert sha256(f)==digest
        if j['phase']=='validation_pilot':continue
        rows.append(dict(variant=j['variant'],horizon=j['horizon'],seed=j['seed'],run=str(run),
            **r['metrics'],**r['details'],**{k:v for k,v in r['diagnostics'].items() if isinstance(v,(int,float))}))
    frame=pd.DataFrame(rows);assert len(frame)==48
    variants={'full':'HDA-NQ (Ours)','fixed':'Fixed offsets','points_4':'K=4','points_12':'K=12',
        'offset_0p5':'Offset=0.5','offset_3':'Offset=3','interior_grid':'Interior grid','calibration':'Calibration loss'}
    stats={}
    for n in variants:
        for h in (96,720):
            f=frame[(frame.variant==n)&(frame.horizon==h)]
            assert len(f)==3 and set(f.seed)=={2024,2025,2026}
            stats[n,h]={k+s:float(f[k].mean() if s=='_mean' else f[k].std(ddof=1))
                for k in f.select_dtypes(include='number') if k not in ('horizon','seed') for s in ('_mean','_std')}
    def pm(g,k,d=5,scale=1):return '$'+f'{g[k+"_mean"]*scale:.{d}f}'+r'\pm'+f'{g[k+"_std"]*scale:.{d}f}'+'$'
    def table(caption,label,header,body,alignment):
        return '\n'.join([r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',
            r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}'+alignment+'@{}}',r'\toprule',
            ' & '.join(header)+r' \\',r'\midrule',*body,r'\bottomrule',r'\end{tabular*}',r'\end{table}'])+'\n'
    body=[]
    for n,label in variants.items():
        if n=='calibration':continue
        body.append(' & '.join([label,*[v for h in (96,720) for v in [pm(stats[n,h],'QuantileLoss_standardized'),pm(stats[n,h],'boundary_rate',2,100)]]])+r' \\')
    tex=r'\section{Additional Sampling and Calibration Studies}'+'\n'
    tex+=('All settings use the final ETTh1 configuration, $L=336$, $H=96/720$, and three seeds. '
        'Sampling count and offset range are varied one at a time. The interior grid spans $[0.5,N-1.5]$ rather than $[0,N-1]$. '
        'Each modified configuration is retrained; full and fixed-offset reference models are reused. Boundary diagnostics average the same first, middle, and last test windows over all variables, steps, heads, and seeds.\n')
    tex+=table('Sampling and boundary comparisons.','tab:reviewer-sampling',
        ['Variant','Pinball 96',r'Boundary 96 (\%)','Pinball 720',r'Boundary 720 (\%)'],body,'lrrrr')
    body=[]
    for h in (96,720):
        for n in ('full','calibration'):
            g=stats[n,h];body.append(' & '.join([str(h),variants[n],pm(g,'QuantileLoss_standardized'),pm(g,'ACE',4),
                pm(g,'PICP90',2,100),pm(g,'PINAW90',4),pm(g,'NormalizedWinkler90',4)])+r' \\')
    selection=json.loads((root/'selection.json').read_text());weight=selection['weight']
    tex+=('The calibration regularizer uses $\\operatorname{mean}_{c,q}(\\widehat c_{cq}-\\tau_q)^2$, '
        'where $\\widehat c_{cq}=\\operatorname{mean}_{b,h}\\sigma((\\widehat Q_{bhcq}-Y_{bhc})/0.1)$. '
        f'The weight $\\lambda={weight:g}$ is selected only on ETTh1 $H=96$, seed 2024 validation, from three eight-epoch pilots, '
        'then frozen across both formal horizons and all seeds. Validation pinball selects checkpoints in every run. '
        'ACE uses hard-threshold quantile coverage averaged separately over variables and quantiles; it is a marginal diagnostic, not a conditional coverage guarantee.\n')
    if not selection['pinball_constraint_satisfied']:
        tex+='No pilot met the predeclared two-percent validation-pinball tolerance; the chosen weight is therefore reported as a calibration--accuracy tradeoff diagnostic.\n'
    tex+=table('Calibration-loss comparison.','tab:reviewer-calibration',
        ['$H$','Variant','Pinball','ACE',r'PICP90 (\%)','PINAW90','Norm. IS90'],body,'llrrrrr')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.65))
    for ax,points,names,label in [
        (axes[0],[4,8,12],['points_4','full','points_12'],'Sampling points K'),
        (axes[1],[0,.5,1.5,3],['fixed','offset_0p5','full','offset_3'],'Maximum offset')]:
        for h,color in [(96,'#237CA2'),(720,'#C3752B')]:
            ax.errorbar(points,[stats[n,h]['QuantileLoss_standardized_mean'] for n in names],
                yerr=[stats[n,h]['QuantileLoss_standardized_std'] for n in names],marker='o',capsize=2,color=color,label=f'H={h}')
        ax.set(xlabel=label,ylabel='Test pinball');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ('pdf','png','svg'):fig.savefig(paper/f'figures/reviewer_sampling.{ext}',bbox_inches='tight',dpi=220)
    plt.close(fig)
    tex+=r'\begin{figure}[htbp]\centering\includegraphics[width=.95\linewidth]{figures/reviewer_sampling.pdf}\caption{Sampling configuration sensitivity on ETTh1.}\label{fig:reviewer-sampling}\end{figure}'+'\n'
    fig,axes=plt.subplots(1,2,figsize=(6.9,2.65))
    for ax,h in zip(axes,(96,720)):
        for n,color in [('full','#237CA2'),('calibration','#C3752B')]:
            coverage=[]
            for row in frame[(frame.variant==n)&(frame.horizon==h)].itertuples():
                with np.load(Path(row.run)/'quantile_calibration.npz') as data:
                    coverage.append(data['coverage'].mean(0));tau=data['quantiles']
            a=np.asarray(coverage);ax.errorbar(tau,a.mean(0),yerr=a.std(0,ddof=1),color=color,marker='o',ms=3,capsize=2,label=variants[n])
        ax.plot([0,1],[0,1],'--',color='#89949A');ax.set(title=f'H={h}',xlabel='Nominal quantile',ylabel='Empirical coverage');ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ('pdf','png','svg'):fig.savefig(paper/f'figures/reviewer_calibration.{ext}',bbox_inches='tight',dpi=220)
    plt.close(fig)
    tex+=r'\begin{figure}[htbp]\centering\includegraphics[width=.95\linewidth]{figures/reviewer_calibration.pdf}\caption{Quantile coverage with and without calibration regularization.}\label{fig:reviewer-calibration}\end{figure}'+'\n'
    (paper/'results/reviewer_supplement.tex').write_text(tex)
    summary=r'\subsection{Additional Sampling and Calibration Studies}'+'\n'
    summary+=('We additionally vary $K\\in\\{4,8,12\\}$ and the offset bound in $\\{0,0.5,1.5,3\\}$ one at a time, '
        'and compare the original reference grid with a half-token interior grid. '
        'All modified settings are retrained on ETTh1 at $H=96/720$ with three seeds. '
        f'A validation-selected calibration regularizer ($\\lambda={weight:g}$, temperature 0.1) is also evaluated; '
        'the ordered head and validation-pinball checkpoint rule are retained. ')
    for h in (96,720):
        a=stats['full',h];b=stats['calibration',h]
        summary+=(f'At $H={h}$, calibration regularization changes mean ACE from {a["ACE_mean"]:.4f} to {b["ACE_mean"]:.4f} '
            f'and pinball from {a["QuantileLoss_standardized_mean"]:.5f} to {b["QuantileLoss_standardized_mean"]:.5f}. ')
    summary+='Complete sensitivity, boundary, coverage, width, and interval-score results are reported in the supplementary study.\n'
    (paper/'results/reviewer_summary.tex').write_text(summary)
    frame.to_csv(paper/'results/reviewer_runs.csv',index=False)
    pd.DataFrame([dict(variant=n,horizon=h,**g) for (n,h),g in stats.items()]).to_csv(paper/'results/reviewer_aggregates.csv',index=False)
    atomic_json(paper/'results/reviewer_audit.json',dict(status='passed',formal_runs=36,pilots=3,reused_evaluations=12,
        source_sha256=sha256(root/'records.json'),selection_sha256=sha256(root/'selection.json')))
    return True
