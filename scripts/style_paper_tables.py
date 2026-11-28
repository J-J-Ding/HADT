"""Consistent model order, best-value emphasis, and first-place counts."""
from pathlib import Path
import pandas as pd

P=Path('HADT_NQ_ICCRD2027_LaTeX/results')
MODELS={'hadt':'HDA-NQ (Ours)','dlinear':'DLinear-NQ','patchtst':'PatchTST-NQ',
        'timemixer':'TimeMixer-NQ','itransformer':'iTransformer-NQ'}


def write_table(name,caption,label,headers,rows,alignment):
    single=name in ('revised_final_ablation.tex','completion_dataset_table.tex')
    text='\n'.join([r'\begin{table*}[t]',r'\centering\small',
        r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'+alignment+'@{}}',
        r'\toprule',' & '.join(headers)+r' \\',r'\midrule',*rows,
        r'\bottomrule',r'\end{tabular*}',r'\end{table*}'])+'\n'
    if single:text=text.replace('table*','table').replace(r'{\textwidth}',r'{\linewidth}')
    (P/name).write_text(text)


def bold(value,enabled):
    return r'\boldsymbol{'+value+'}' if enabled else value


def format_tables(runs_path, aggregates_path):
    runs=pd.read_csv(runs_path);agg=pd.read_csv(aggregates_path)
    datasets=[d for d in ('ETTh1','ETTh2','ETTm1','ETTm2','Traffic','Weather','Electricity') if d in set(runs.dataset)];rows=[];counts={m:0 for m in MODELS}
    for d in datasets:
        for h in sorted(runs.loc[runs.dataset==d,'horizon'].unique()):
            g=agg[(agg.dataset==d)&(agg.horizon==h)].set_index('model')
            assert set(g.index)==set(MODELS)
            best=g.QuantileLoss_standardized_mean.min();cells=[d,str(h)]
            for m in MODELS:
                r=g.loc[m];win=r.QuantileLoss_standardized_mean==best;counts[m]+=int(win)
                value=f'{r.QuantileLoss_standardized_mean:.5f}'+r'\pm'+f'{r.QuantileLoss_standardized_std:.5f}'
                cells.append('$'+bold(value,win)+'$')
            rows.append(' & '.join(cells)+r' \\')
    rows.extend([r'\midrule',' & '.join([r'\multicolumn{2}{l}{\textbf{1st Count}}',*[str(counts[m]) for m in MODELS]])+r' \\'])
    write_table('revised_external.tex','Completed common-head comparisons: test pinball mean $\\pm$ sample SD over three seeds, with $L=336$. '
                'Every row contains all five models and three seeds. Bold marks the lowest mean; 1st Count counts wins using unrounded means, including ties. '
                'Settings are included by completion, not performance.',
                'tab:external',['Dataset','$H$',*MODELS.values()],rows,'llrrrrr')
    rows=[];coverage_counts={m:0 for m in MODELS};score_counts={m:0 for m in MODELS}
    intervalstats=[]
    for d in datasets:
        frame=runs[runs.dataset==d];hs=sorted(frame.horizon.unique())
        g=frame.groupby(['name','seed'])[['PICP90','PINAW90','NormalizedWinkler90']].mean().groupby('name').agg(['mean','std'])
        best_coverage=(g['PICP90']['mean']-.9).abs().min();best_score=g['NormalizedWinkler90']['mean'].min()
        cells=[d,','.join(map(str,hs))]
        for m in MODELS:
            picp=g.loc[m,('PICP90','mean')];score=g.loc[m,('NormalizedWinkler90','mean')]
            cw=abs(picp-.9)==best_coverage;sw=score==best_score
            coverage_counts[m]+=int(cw);score_counts[m]+=int(sw)
            cells.append('$'+bold(f'{100*picp:.2f}',cw)+r'\,/\,'+bold(f'{score:.4f}',sw)+'$')
            intervalstats.append((d,m,g.loc[m],cw,sw))
        rows.append(' & '.join(cells)+r' \\')
    rows.extend([r'\midrule',' & '.join([r'\multicolumn{2}{l}{\textbf{1st Count}}',
                *['$'+str(coverage_counts[m])+r'\,/\,'+str(score_counts[m])+'$' for m in MODELS]])+r' \\'])
    write_table('revised_external_intervals.tex','Interval results: PICP90 (\\%) / normalized IS90, averaged equally over the listed horizons within seed and then over three seeds. '
                'Bold marks coverage closest to 90\\% and the lowest interval score separately. '
                '1st Count reports coverage wins / score wins across datasets; ties count for every tied model. Widths and variability appear in Supplementary Table S3.',
                'tab:external-intervals',['Dataset','$H$',*MODELS.values()],rows,'llrrrrr')
    # These tables have variants in rows rather than model columns.
    ab=pd.read_csv(P/'completion_ablation_aggregates.csv')
    labels={'full':'HDA-NQ (Ours)','fixed':'Fixed offsets','shared':'Shared embedding','dense':'Dense attention',
            'no_skip':'No correction','fixed_no_skip':'Fixed + no correction','independent':'Independent head'}
    g=ab[ab.horizon==96].set_index('variant');best=g.QuantileLoss_standardized_mean.min();rows=[]
    for n,label in labels.items():
        r=g.loc[n];value=f'{r.QuantileLoss_standardized_mean:.5f}'+r'\pm'+f'{r.QuantileLoss_standardized_std:.5f}'
        crossing=r.CrossingRate_mean*100
        if crossing==0:cross=r'\boldsymbol{0}'
        else:
            a,b=f'{crossing:.2e}'.split('e');cross=a+r'\times10^{'+str(int(b))+'}'
        rows.append(' & '.join([label,'$'+bold(value,r.QuantileLoss_standardized_mean==best)+'$','$'+cross+'$'])+r' \\')
    write_table('revised_final_ablation.tex','ETTh1 component study at $H=96$ (long-term forecasting). Mean $\\pm$ sample SD over three seeds; crossing is the mean percentage. '
                'Bold marks the lowest loss or crossing rate, including ties. Complete $H=96/720$ results appear in Supplementary Table S1.',
                'tab:final-ablation',['Variant','Pinball',r'Crossing (\%)'],rows,'lrr')
    p=P/'completion_dataset_table.tex';s=p.read_text().replace(r'\begin{table}[t]',r'\begin{table*}[t]').replace(r'\end{table}',r'\end{table*}')
    s=s.replace(r'\begin{tabular}{@{}lrrl@{}}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrl@{}}').replace(r'\end{tabular}',r'\end{tabular*}')
    p.write_text(s.replace('table*','table').replace(r'{\textwidth}',r'{\linewidth}'))
    # Supplement: keep Ours first and emphasize accuracy/coverage/score by dataset.
    rows=[]
    for d,m,r,cw,sw in intervalstats:
        cells=[d,MODELS[m]]
        for key,win,scale,digits in [('PICP90',cw,100,2),('PINAW90',False,1,4),('NormalizedWinkler90',sw,1,4)]:
            value=f'{r[(key,"mean")]*scale:.{digits}f}'+r'\pm'+f'{r[(key,"std")]*scale:.{digits}f}'
            cells.append('$'+bold(value,win)+'$')
        rows.append(' & '.join(cells)+r' \\')
    # The supplement itself is single-column; its tables use that full text width.
    write_table('supplement_external_intervals.tex','Interval statistics for the same completed settings as the main table. '
                'Mean $\\pm$ sample SD over three seeds after averaging horizons within seed. Bold follows the main interval table.',
                'tab:full-intervals',['Dataset','Model',r'PICP90 (\%)','PINAW90','Norm. IS90'],rows,'llrrr')
    p=P/'supplement_external_intervals.tex';p.write_text(p.read_text().replace('table*','table'))
    return dict(pinball=counts,coverage=coverage_counts,interval_score=score_counts)


if __name__=='__main__':
    import json
    result=format_tables(P/'snapshot_baseline_runs.csv',P/'snapshot_baseline_aggregates.csv')
    (P/'first_place_counts.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
