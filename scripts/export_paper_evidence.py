"""Build manuscript evidence exclusively from completed, traceable experiment records."""
import hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

ROOT=Path('outputs/paper_study')
PAPER=Path('HADT_NQ_ICCRD2027_LaTeX')
SEEDS={2024,2025,2026}
LABELS={'hadt':'HADT-NQ','linear':'Linear-NQ','patch':'Patch-NQ','dense':'Dense-NQ','fixed':'Fixed-NQ',
        'shared':'Shared query','independent':'Independent head','no_revin':'Without RevIN'}
METRICS=['QuantileLoss_standardized','PICP50','PICP80','PICP90','PINAW50','PINAW80','PINAW90',
         'NormalizedWinkler50','NormalizedWinkler80','NormalizedWinkler90','CrossingRate','best_val_pinball']


def unique_records(records):
    found={}
    for original in records:
        r=dict(original)
        key=(r['dataset'],r['method'],int(r['horizon']),int(r['seed']))
        if key in found:
            for m in METRICS:
                if m in r and m in found[key] and not np.isclose(r[m],found[key][m],rtol=0,atol=1e-12):
                    raise ValueError(f'Conflicting repeated measurement {key}: {m}')
        if 'prior_run' in r.get('source_kind',''):
            r['code_sha256']=None
            r['provenance_note']='Historical source digest not captured; matched configuration, dataset digest and checkpoint origin retained.'
        if r['method']=='independent':
            r['interval_score_scope']='Raw endpoint diagnostics only; paper reports pinball and crossing, not proper interval comparisons.'
        found[key]=r
    return list(found.values())


def aggregate_complete_groups(records):
    records=unique_records(records)
    groups={}
    for r in records:groups.setdefault((r['dataset'],r['method'],r['horizon']),[]).append(r)
    rows=[]
    for (dataset,method,horizon),group in sorted(groups.items()):
        if {int(r['seed']) for r in group}!=SEEDS:continue
        row=dict(dataset=dataset,method=method,horizon=horizon,n=len(group))
        for metric in METRICS:
            if method=='independent' and metric.startswith(('PICP','PINAW','NormalizedWinkler')):
                continue
            if all(metric in r for r in group):
                a=np.asarray([r[metric] for r in group],dtype=float)
                if not np.isfinite(a).all():raise ValueError(f'Nonfinite {dataset}/{method}/{metric}')
                row[metric+'_mean']=a.mean();row[metric+'_std']=a.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def load_records():
    rows=[]
    for name in ['records.json','traffic_case_records.json','traffic_dense_records.json','traffic_repeat_records.json']:
        p=ROOT/name
        if p.exists():rows.extend(json.loads(p.read_text()))
    for p in ROOT.glob('*/*/H*/seed*/complete.json'):
        rows.append(json.loads(p.read_text()))
    return unique_records(rows)


def audit_provenance(records):
    audit=[]
    for r in records:
        path=Path(r['source_dir']);marker=path/'provenance.json'
        entry={k:r[k] for k in ('dataset','method','horizon','seed','source_dir','source_kind')}
        if marker.exists():
            raw=json.loads(marker.read_text());payload={k:v for k,v in raw.items() if k!='fingerprint'}
            # v1 signed pre-fit config; the original writer retained a mutable config reference.
            # Remove only the two documented runtime additions, retain the raw marker for inspection.
            payload['config'].pop('data_scaler',None);payload['config'].pop('feature_columns',None)
            value=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            if value!=raw['fingerprint']:raise ValueError(f'Invalid canonical provenance: {marker}')
            canonical=dict(fingerprint=value,signed_payload=payload,
                           normalization='Pre-fit configuration: exclude runtime scaler/feature-column additions; resolved_config.json retains them.')
            (path/'canonical_provenance.json').write_text(json.dumps(canonical,indent=2))
            entry.update(status='canonical pre-fit fingerprint verified',code_sha256=payload['code_sha256'])
        else:
            entry.update(status='historical configuration/data/checkpoint import; original code digest unavailable',code_sha256=None)
        audit.append(entry)
    (PAPER/'results/provenance_audit.json').write_text(json.dumps(audit,indent=2))
    return audit


def pm(row,metric,digits=4,percent=False):
    scale=100 if percent else 1
    return f"${row[metric+'_mean']*scale:.{digits}f}\\,\\pm\\,{row[metric+'_std']*scale:.{digits}f}$"


def lookup(frame,dataset,method,horizon=96):
    if frame.empty:return None
    selected=frame[(frame.dataset==dataset)&(frame.method==method)&(frame.horizon==horizon)]
    return None if selected.empty else selected.iloc[0]


def table_file(path,caption,label,header,rows,wide=True):
    env='table*' if wide else 'table'
    alignment='ll'+'r'*(len(header)-2)
    text=[f'\\begin{{{env}}}[t]',f'\\caption{{{caption}}}',f'\\label{{{label}}}',
          '\\centering','\\small',f'\\begin{{tabular}}{{@{{}}{alignment}@{{}}}}','\\toprule',
          ' & '.join(header)+r' \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',f'\\end{{{env}}}']
    path.write_text('\n'.join(text)+'\n')


def generate_tables(records,frame):
    out=PAPER/'results';out.mkdir(exist_ok=True)
    rows=[]
    for dataset in ['ETTh1','ETTh2']:
        for method in ['hadt','linear','patch','dense','fixed']:
            r=lookup(frame,dataset,method)
            if r is not None:
                rows.append(' & '.join([dataset,LABELS[method],pm(r,'QuantileLoss_standardized',5),pm(r,'PICP90',2,True),pm(r,'PINAW90',4),pm(r,'NormalizedWinkler90',4)])+r' \\')
    table_file(out/'table_main.tex','Primary comparison at $H=96$. Values are mean $\\pm$ sample standard deviation over seeds 2024, 2025, and 2026. PICP is reported in percent; its spread is in percentage points. Lower pinball, width, and interval score are preferable, but width must be read with coverage. Controls are local quantile adaptations, not official point-forecasting reproductions. Only complete three-seed groups enter this table.',
               'tab:main',['Dataset','Method','Pinball $\\downarrow$','PICP90 (\\%)','PINAW90','Norm. Winkler90 $\\downarrow$'],rows)
    ab=[]
    for method in ['hadt','fixed','dense','shared','independent','no_revin']:
        r=lookup(frame,'ETTh1',method)
        if r is not None:ab.append(' & '.join(['ETTh1',LABELS[method],pm(r,'QuantileLoss_standardized',5),pm(r,'CrossingRate',3,True)])+r' \\')
    table_file(out/'table_ablation.tex','ETTh1 component comparisons at $H=96$, three seeds. Crossing is the percentage of variable--horizon forecasts with any adjacent quantile-order violation. Independent-head interval scores are omitted because unordered endpoints do not define valid central intervals.',
               'tab:ablation',['Dataset','Variant','Pinball $\\downarrow$','Crossing (\\%)'],ab,wide=False)
    extension=[]
    for dataset in ['ETTh1','ETTh2']:
        for method in ['hadt','linear','dense']:
            r=lookup(frame,dataset,method,192)
            if r is not None:extension.append(' & '.join([dataset,LABELS[method],pm(r,'QuantileLoss_standardized',5),pm(r,'PICP90',2,True),pm(r,'PINAW90',4)])+r' \\')
    if extension:
        table_file(out/'table_horizon.tex','Completed three-seed extension at $H=192$, using the same history length and training policy. This extension compares HADT-NQ, Linear-NQ, and Dense-NQ; it does not claim an exhaustive horizon sweep.',
                   'tab:horizon',['Dataset','Method','Pinball $\\downarrow$','PICP90 (\\%)','PINAW90'],extension)
    else:(out/'table_horizon.tex').write_text('')
    case=[]
    for method in ['hadt','linear','dense']:
        a=[r for r in records if r['dataset']=='Traffic' and r['method']==method and r['horizon']==96 and r['seed']==2024]
        if a:
            r=a[0];case.append(' & '.join(['Traffic',LABELS[method],f"{r['QuantileLoss_standardized']:.5f}",f"{r['PICP90']*100:.2f}",f"{r['PINAW90']:.5f}"])+r' \\')
    table_file(out/'table_traffic.tex','High-dimensional Traffic case at $H=96$, seed 2024 only, with all 862 variables evaluated. These are individual runs, not three-seed means or evidence of seed stability.',
               'tab:traffic',['Dataset','Method','Pinball $\\downarrow$','PICP90 (\\%)','PINAW90'],case)
    paragraphs=[]
    for dataset in ['ETTh1','ETTh2']:
        h=lookup(frame,dataset,'hadt');p=lookup(frame,dataset,'patch');l=lookup(frame,dataset,'linear');d=lookup(frame,dataset,'dense');f=lookup(frame,dataset,'fixed')
        if h is None:continue
        sentence=f"On {dataset}, HADT-NQ obtains mean pinball {h['QuantileLoss_standardized_mean']:.5f} and 90\\% interval coverage {h['PICP90_mean']*100:.2f}\\%."
        if p is not None:sentence+=f" Patch-NQ obtains {p['QuantileLoss_standardized_mean']:.5f};"
        if l is not None:sentence+=f" Linear-NQ obtains {l['QuantileLoss_standardized_mean']:.5f}."
        if d is not None:
            delta=(h['QuantileLoss_standardized_mean']/d['QuantileLoss_standardized_mean']-1)*100
            sentence+=f" Relative to Dense-NQ, HADT-NQ's mean loss changes by {delta:+.2f}\\% (negative denotes improvement)."
        if f is not None:
            delta=(h['QuantileLoss_standardized_mean']/f['QuantileLoss_standardized_mean']-1)*100
            sentence+=f" Relative to Fixed-NQ, the change is {delta:+.2f}\\%."
        if dataset=='ETTh1' and f is not None:
            sentence+=f" The normalized 90\\% interval score is {h['NormalizedWinkler90_mean']:.4f} for HADT-NQ versus {f['NormalizedWinkler90_mean']:.4f} for Fixed-NQ; its pinball advantage therefore does not extend uniformly to interval-score quality."
        paragraphs.append(sentence+'\n')
    (out/'summary_main.tex').write_text('\n\n'.join(paragraphs)+'\n')
    paragraphs=[]
    for method in ['shared','independent','no_revin']:
        r=lookup(frame,'ETTh1',method)
        if r is not None:
            text=f"{LABELS[method]} obtains mean pinball {r['QuantileLoss_standardized_mean']:.5f}"
            if method=='independent':text+=f" and crossing rate {100*r['CrossingRate_mean']:.3f}\\%. Its raw endpoint widths and interval scores are not interpreted as valid proper-interval comparisons."
            else:text+='.'
            paragraphs.append(text)
    (out/'summary_ablation.tex').write_text(' '.join(paragraphs)+'\n')
    abstract=[]
    contrasts=[];linear_lower=[]
    for dataset in ['ETTh1','ETTh2']:
        for horizon in [96,192]:
            h=lookup(frame,dataset,'hadt',horizon);d=lookup(frame,dataset,'dense',horizon);l=lookup(frame,dataset,'linear',horizon)
            if all(x is not None for x in [h,d,l]):
                contrasts.append(100*(1-h['QuantileLoss_standardized_mean']/d['QuantileLoss_standardized_mean']))
                linear_lower.append(l['QuantileLoss_standardized_mean']<h['QuantileLoss_standardized_mean'])
    if len(contrasts)==4 and min(contrasts)>0 and all(linear_lower):
        abstract.append(f"Across four ETT dataset--horizon settings, HADT-NQ reduces mean pinball relative to Dense-NQ by {min(contrasts):.2f}--{max(contrasts):.2f}\\%, yet Linear-NQ achieves lower loss in all four.")
    traffic={r['method']:r for r in records if r['dataset']=='Traffic' and r['horizon']==96 and r['seed']==2024}
    if 'dense' in traffic and 'hadt' in traffic:
        abstract.append("Dense-NQ is stronger in the single-seed Traffic case, indicating that sparse retrieval is not uniformly preferable.")
    (out/'abstract_results.tex').write_text(' '.join(abstract)+'\n')
    traffic_text=[]
    traffic={r['method']:r for r in records if r['dataset']=='Traffic' and r['horizon']==96 and r['seed']==2024}
    if 'hadt' in traffic and 'linear' in traffic:
        h=traffic['hadt'];l=traffic['linear']
        traffic_text.append(f"In this single run, HADT-NQ obtains pinball {h['QuantileLoss_standardized']:.5f}, compared with {l['QuantileLoss_standardized']:.5f} for Linear-NQ.")
    if 'dense' in traffic:
        d=traffic['dense']
        traffic_text.append(f"Dense-NQ obtains {d['QuantileLoss_standardized']:.5f}; it has the lower loss in this Traffic case, whereas HADT-NQ has lower mean loss than Dense-NQ in the repeated ETT comparisons. Thus, sparse retrieval is not uniformly preferable.")
    (out/'summary_traffic.tex').write_text(' '.join(traffic_text)+'\n')


def sample_path(record,name='forecast_samples.npz'):
    p=Path(record['source_dir'])/name
    return p if p.exists() else Path(record['source_dir'])/'figures'/name


def generate_figures(records,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':9,
                         'axes.titlesize':9,'axes.labelsize':9,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8})
    out=PAPER/'figures';out.mkdir(exist_ok=True)
    plot_data=PAPER/'results/figure_data';plot_data.mkdir(parents=True,exist_ok=True)
    calibration_data=[];forecast_data={}
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
        fig.savefig(out/f'{name}.png',dpi=220,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.45),sharex=True,sharey=True)
    colors={'hadt':'#246a9b','patch':'#c98532','linear':'#778344'}
    styles={'hadt':('o','-'),'patch':('s','--'),'linear':('^',':')}
    for ax,dataset in zip(axes,['ETTh1','ETTh2']):
        for method in ['hadt','patch','linear']:
            r=lookup(frame,dataset,method)
            if r is None:continue
            y=np.array([r[f'PICP{v}_mean'] for v in [50,80,90]])*100
            sd=np.array([r[f'PICP{v}_std'] for v in [50,80,90]])*100
            for nominal,mean,spread in zip([50,80,90],y,sd):
                calibration_data.append(dict(dataset=dataset,method=method,nominal_percent=nominal,coverage_percent=mean,sample_sd_percentage_points=spread,n=3))
            marker,line=styles[method]
            ax.errorbar([50,80,90],y,yerr=sd,marker=marker,linestyle=line,color=colors[method],capsize=2,label=LABELS[method])
        ax.plot([40,100],[40,100],color='.4',linestyle='--',linewidth=.8,label='Ideal')
        ax.set(title=dataset,xlabel='Nominal coverage (%)',ylabel='Observed coverage (%)',xlim=(45,95),ylim=(35,100))
        ax.set_xticks([50,80,90]);ax.grid(alpha=.15);ax.legend(loc='upper left')
    save(fig,'paper_calibration')
    pd.DataFrame(calibration_data).to_csv(plot_data/'calibration.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.35))
    for ax,dataset,seed in zip(axes,['ETTh1','Traffic'],[2025,2024]):
        candidates=[r for r in records if r['dataset']==dataset and r['method']=='hadt' and r['horizon']==96 and r['seed']==seed]
        if not candidates:continue
        r=candidates[0];data=dict(np.load(sample_path(r)));c=list(data['feature_columns']).index('OT')
        q=data['quantiles'][0,:,c];y=data['target'][0,:,c];t=np.arange(1,len(y)+1)
        forecast_data[dataset+'_target']=y
        forecast_data[dataset+'_hadt_quantiles']=q
        forecast_data[dataset+'_history']=data['history'][0,:,c]
        ax.fill_between(t,q[:,0],q[:,6],color='#c4daeb',label='HADT 90% interval')
        ax.plot(t,y,color='.15',linewidth=1.1,label='Observed')
        ax.plot(t,q[:,3],color='#246a9b',linewidth=1.1,label='HADT median')
        control='patch' if dataset=='ETTh1' else 'linear'
        matches=[a for a in records if a['dataset']==dataset and a['method']==control and a['horizon']==96 and a['seed']==seed]
        if matches:
            f=dict(np.load(sample_path(matches[0])));j=list(f['feature_columns']).index('OT')
            np.testing.assert_allclose(f['target'][0,:,j],y,atol=1e-7)
            forecast_data[dataset+'_'+control+'_quantiles']=f['quantiles'][0,:,j]
            ax.plot(t,f['quantiles'][0,:,j,3],color=colors[control],linestyle='--',linewidth=1.,label=LABELS[control]+' median')
        ax.set(title=dataset+' | first fixed test window',xlabel='Forecast step',ylabel='Original units')
        ax.grid(alpha=.15);ax.legend(loc='best')
    save(fig,'paper_forecasts')
    np.savez_compressed(plot_data/'forecast_windows.npz',**forecast_data)
    # Historical depth pilot: single seed, same early stopping policy; no uncertainty invented.
    ett_source=Path('outputs/ETTh1_depth_comparison/comparison.csv')
    traffic_source=Path('outputs/Traffic_depth_comparison/comparison.csv')
    if not ett_source.exists():ett_source=PAPER/'results/depth_pilot_etth1.csv'
    if not traffic_source.exists():traffic_source=PAPER/'results/depth_pilot_traffic.csv'
    ett=pd.read_csv(ett_source)
    traffic=pd.read_csv(traffic_source)
    fig,ax=plt.subplots(figsize=(3.4,2.15))
    for data,label,color,marker in [(ett,'ETTh1','#246a9b','o'),(traffic,'Traffic','#c98532','s')]:
        baseline=float(data.loc[data.layers==0,'test_pinball'].iloc[0])
        ax.plot(data.layers.to_numpy(),(data.test_pinball.to_numpy()/baseline-1)*100,marker=marker,color=color,label=label)
    ax.axhline(0,color='.4',linestyle='--',linewidth=.8)
    ax.set(xlabel='Encoder depth (0 = bypass)',ylabel='Test loss change vs. depth 0 (%)')
    ax.set_xticks([0,1,2,3]);ax.legend();ax.grid(alpha=.15)
    save(fig,'paper_depth_pilot')
    import shutil
    for source,name in [(ett_source,'depth_pilot_etth1.csv'),(traffic_source,'depth_pilot_traffic.csv')]:
        target=PAPER/'results'/name
        if source.resolve()!=target.resolve():shutil.copyfile(source,target)
    sources={
      'paper_architecture':{'kind':'schematic','source':'scripts/draw_paper_architecture.py'},
      'paper_calibration':{'kind':'seed mean and sample SD','source':'results/aggregate_metrics.csv','datasets':['ETTh1','ETTh2'],'horizon':96,'methods':['hadt','patch','linear']},
      'paper_forecasts':{'kind':'fixed first test window','source_records':[{'dataset':'ETTh1','method':'hadt','seed':2025,'control':'patch'},{'dataset':'Traffic','method':'hadt','seed':2024,'control':'linear'}],'horizon':96,'variable':'OT'},
      'paper_depth_pilot':{'kind':'single-seed pilot relative to depth0','seed':2024,'source':['results/depth_pilot_etth1.csv','results/depth_pilot_traffic.csv']},
      'paper_sampling':{'kind':'learned intermediate arrays','dataset':'ETTh1','method':'hadt','horizon':96,'seed':2025,'sample':0,'variable':'OT','head':0}}
    (PAPER/'results/figure_sources.json').write_text(json.dumps(sources,indent=2))
    hadt=[r for r in records if r['dataset']=='ETTh1' and r['method']=='hadt' and r['horizon']==96 and r['seed']==2025]
    if hadt:
        aux=np.load(sample_path(hadt[0],'intermediate_steps.npz'));sample=np.load(sample_path(hadt[0]))
        c=list(sample['feature_columns']).index('OT')
        offsets=aux['offsets'][0,c,:,0,:].T
        attention=aux['attention'][0,c,:,0,:].T
        np.savez_compressed(plot_data/'sampling_head0.npz',offsets=offsets,attention=attention,positions=aux['positions'][0,c,:,0,:].T,forecast_steps=np.arange(1,97))
        fig,axes=plt.subplots(2,1,figsize=(3.4,2.75),sharex=True)
        im=axes[0].imshow(offsets,origin='lower',aspect='auto',cmap='coolwarm',vmin=-1.5,vmax=1.5,extent=[1,96,-.5,7.5])
        axes[0].set(ylabel='Sample index',title='Head 0 learned offsets (patch units)');fig.colorbar(im,ax=axes[0],fraction=.04,pad=.02)
        im=axes[1].imshow(attention,origin='lower',aspect='auto',cmap='viridis',vmin=0,vmax=max(.25,attention.max()),extent=[1,96,-.5,7.5])
        axes[1].set(xlabel='Forecast step',ylabel='Sample index',title='Head 0 attention weights');fig.colorbar(im,ax=axes[1],fraction=.04,pad=.02)
        save(fig,'paper_sampling')



def sensitivity_tables(primary):
    source=Path('outputs/paper_sensitivity/records.json')
    rows=[];records=[]
    base=lookup(primary,'ETTh1','hadt')
    if base is not None:rows.append((8,base))
    if source.exists():
        records=json.loads(source.read_text())
        tagged=[dict(r,method='hadt_K'+str(r['sensitivity_K'])) for r in records]
        sf=aggregate_complete_groups(tagged)
        sf.to_csv(PAPER/'results/sampling_sensitivity_aggregates.csv',index=False)
        (PAPER/'results/sampling_sensitivity_records.json').write_text(json.dumps(records,indent=2))
        for k in [4,12]:
            a=lookup(sf,'ETTh1','hadt_K'+str(k))
            if a is not None:rows.append((k,a))
    body=[str(k)+' & '+pm(r,'QuantileLoss_standardized',5)+' & '+pm(r,'PICP90',2,True)+r' \\' for k,r in sorted(rows)]
    table_file(PAPER/'results/table_sampling_sensitivity.tex',
               'ETTh1 sampling sensitivity at $H=96$: mean $\\pm$ sample standard deviation over three seeds. The primary default $K=8$ is retained independently of these exploratory comparisons; offset bounds and training policy are unchanged.',
               'tab:sampling_sensitivity',['$K$','Pinball $\\downarrow$','PICP90 (\\%)'],body,wide=False)
    return records


def main():
    (PAPER/'results').mkdir(exist_ok=True)
    records=load_records()
    frame=aggregate_complete_groups(records)
    (PAPER/'results/evidence_records.json').write_text(json.dumps(records,indent=2))
    pd.DataFrame(records).to_csv(PAPER/'results/evidence_runs.csv',index=False)
    frame.to_csv(PAPER/'results/aggregate_metrics.csv',index=False)
    (PAPER/'results/seed_policy.json').write_text(json.dumps(dict(required=sorted(SEEDS),ddof=1,
            complete_groups=len(frame),completed_runs=len(records),traffic_case_seed=2024,
            historical_source_digests='unknown; not represented as freshly captured'),indent=2))
    generate_tables(records,frame);generate_figures(records,frame)
    sensitivity=sensitivity_tables(frame)
    audit_provenance(records+sensitivity)
    print('Exported',len(records),'completed runs and',len(frame),'complete three-seed groups.')
    if not frame.empty:print(frame[['dataset','method','horizon','n','QuantileLoss_standardized_mean','QuantileLoss_standardized_std']].to_string(index=False))


if __name__=='__main__':main()
