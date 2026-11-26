"""Build transparent reports from measured search/confirmation/test records."""
import argparse
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from experiments.detail_search import candidates,configuration,HORIZONS,SEEDS,scores
from utils.provenance import atomic_json,sha256,source_manifest

plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})


def flatten(records,test=False):
    rows=[]
    for r in records:
        row={k:r[k] for k in ('candidate','history','horizon','seed','epochs_run','best_epoch','train_val_gap')}
        row.update(r['test_detail'] if test else r['validation'])
        if test:
            row['NormalizedWinkler90']=r['test']['NormalizedWinkler90']
            row['CrossingRate']=r['test']['CrossingRate']
        row['source_dir']=r['source_dir'];rows.append(row)
    return pd.DataFrame(rows)


def aggregate(frame):
    metrics=['pinball','median_mae','delta_mae','jump_mae','roughness_ratio','PICP90','PINAW90','train_val_gap','epochs_run','best_epoch']
    metrics += [k for k in ('NormalizedWinkler90','CrossingRate') if k in frame]
    return frame.groupby(['candidate','history','horizon'])[metrics].agg(['mean','std']).reset_index()


def save(fig,root,name):
    fig.tight_layout();fig.savefig(root/(name+'.png'),dpi=160,bbox_inches='tight')
    fig.savefig(root/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def format_table(frame):
    return frame.to_html(index=False,float_format=lambda x:f'{x:.6f}',escape=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/etth1_detail_search'))
    args=parser.parse_args();root=args.root
    protocol=json.loads((root/'protocol.json').read_text())
    selection=json.loads((root/'selection.json').read_text())
    by_id={c['id']:c for c in candidates()}
    confirmed=json.loads((root/'confirm_records.json').read_text())
    tests=json.loads((root/'test_records.json').read_text())
    for r in confirmed:
        for p,h in r['artifact_sha256'].items():assert sha256(Path(r['source_dir'])/p)==h
        assert r['history']==selection['history'] and r['test_used'] is False
    for r in tests:
        assert r['test_used_for_selection'] is False
        assert r['signature']['selection_sha256']==sha256(root/'selection.json')
        if (root/'horizon_selection.json').exists():assert r['signature']['horizon_selection_sha256']==sha256(root/'horizon_selection.json')
        for p,h in r['artifact_sha256'].items():assert sha256(Path(r['source_dir'])/p)==h
    assert source_manifest()==protocol['source']
    val=flatten(confirmed);test=flatten(tests,True)
    val.to_csv(root/'validation_metrics.csv',index=False);test.to_csv(root/'test_detail_metrics.csv',index=False)
    for frame,label in [(val,'validation'),(test,'test')]:
        agg=aggregate(frame);agg.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in agg.columns]
        agg.to_csv(root/(label+'_aggregates.csv'),index=False)
    ranking=pd.DataFrame(selection['ranking']);ranking.to_csv(root/'confirmation_ranking.csv',index=False)
    chosen=list(dict.fromkeys(['reference']+sorted({r['candidate'] for r in tests})))
    testagg=pd.read_csv(root/'test_aggregates.csv')
    for c in chosen:
        for h in HORIZONS:
            group=test[(test.candidate==c)&(test.horizon==h)]
            assert set(group.seed)==set(SEEDS) and len(group)==3
    # Export a standalone deployable config without changing historical defaults.
    config=configuration(by_id[selection['recommended']],selection['history'],30,'optimized_etth1')
    config['visualization']=dict(enabled=True,max_channels=7)
    config['experiment'].update(run_name='optimized_etth1',horizons=list(HORIZONS),seeds=list(SEEDS))
    config['optimization']=dict(selection_source=str(root/'selection.json'),test_used_for_selection=False,
        candidate=by_id[selection['recommended']],note='Use scripts/retrain_etth1_selected.py to reproduce the declared training objective, including optional difference supervision.')
    Path('configs/optimized_etth1.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    horizon_choice=json.loads((root/'horizon_selection.json').read_text()) if (root/'horizon_selection.json').exists() else None
    if horizon_choice:
        for h in HORIZONS:
            candidate=by_id[horizon_choice['choices'][str(h)]['detail']]
            hc=configuration(candidate,selection['history'],30,'optimized_etth1')
            hc['experiment'].update(run_name=f'optimized_etth1_H{h}',horizons=[h])
            hc['visualization']=dict(enabled=True,max_channels=7)
            hc['optimization']=dict(selection_source=str(root/'horizon_selection.json'),candidate=candidate,test_used_for_selection=False)
            Path(f'configs/optimized_etth1_H{h}.yaml').write_text(yaml.safe_dump(hc,sort_keys=False))
            ac=configuration(by_id[horizon_choice['choices'][str(h)]['accuracy']],selection['history'],30,'optimized_etth1')
            ac['experiment'].update(run_name=f'optimized_etth1_H{h}_accuracy',horizons=[h])
            ac['visualization']=dict(enabled=True,max_channels=7)
            ac['optimization']=dict(selection_source=str(root/'horizon_selection.json'),candidate=by_id[horizon_choice['choices'][str(h)]['accuracy']],test_used_for_selection=False)
            Path(f'configs/optimized_etth1_H{h}_accuracy.yaml').write_text(yaml.safe_dump(ac,sort_keys=False))
    for role in ('accuracy','detail'):
        c=configuration(by_id[selection[role]],selection['history'],30,'optimized_etth1')
        c['visualization']=dict(enabled=True,max_channels=7)
        c['experiment']['run_name']='optimized_etth1_'+role
        c['optimization']=dict(selection_source=str(root/'selection.json'),candidate=by_id[selection[role]],test_used_for_selection=False)
        Path(f'configs/optimized_etth1_{role}.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    # Matched learning curves: seed2024, all horizons, all tested configurations.
    fig,axes=plt.subplots(len(chosen),4,figsize=(18,3.4*len(chosen)),squeeze=False)
    for row,c in enumerate(chosen):
        for col,h in enumerate(HORIZONS):
            record=next(r for r in confirmed if r['candidate']==c and r['horizon']==h and r['seed']==2024)
            history=pd.read_csv(Path(record['source_dir'])/'training_history.csv')
            ax=axes[row,col]
            ax.plot(history.epoch.to_numpy(),history.train_pinball.to_numpy(),label='Train objective')
            ax.plot(history.epoch.to_numpy(),history.val_pinball.to_numpy(),label='Validation pinball')
            ax.axvline(record['best_epoch'],ls='--',color='gray',lw=.7)
            ax.set(title=f'{c} | H{h}',xlabel='Epoch',ylabel='Loss');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Validation-selected checkpoints; train objective may include the declared difference penalty')
    save(fig,root,'learning_curves')
    # Detail diagnostics over all horizons; roughness is descriptive, not a target.
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for c in chosen:
        sub=testagg[testagg.candidate==c].sort_values('horizon')
        for ax,key in zip(axes,['pinball','delta_mae','jump_mae']):
            ax.errorbar(sub.horizon.to_numpy(),sub[key+'_mean'].to_numpy(),yerr=sub[key+'_std'].to_numpy(),marker='o',capsize=3,label=c)
            ax.set(xlabel='Forecast horizon',ylabel=key);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);save(fig,root,'metric_comparison')
    # Fixed first/middle/last origins; first 96 forecast steps shown for local inspection.
    for h in HORIZONS:
        data={}
        for c in chosen:
            r=next(r for r in tests if r['candidate']==c and r['horizon']==h and r['seed']==2024)
            with np.load(Path(r['source_dir'])/'figures/forecast_samples.npz') as f:data[c]={k:f[k] for k in f.files}
        ref=data['reference'];names=[str(x) for x in ref['feature_columns']]
        selected=['OT','HUFL','MUFL'];indices=[names.index(x) for x in selected if x in names]
        fig,axes=plt.subplots(3,len(indices),figsize=(5*len(indices),9),squeeze=False)
        colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(3):
            for j,ch in enumerate(indices):
                ax=axes[i,j];length=min(96,h);t=np.arange(length)
                ax.plot(t,ref['target'][i,:length,ch],color='black',lw=1.2,label='Observed future')
                for k,c in enumerate(chosen):
                    assert np.array_equal(data[c]['sample_indices'],ref['sample_indices'])
                    np.testing.assert_allclose(data[c]['target'],ref['target'],atol=0,rtol=0)
                    ax.plot(t,data[c]['quantiles'][i,:length,ch,3],lw=1,label=c,color=colors[k])
                ax.set(title=f'H{h} | {names[ch]} | origin {ref["sample_indices"][i]}',xlabel='Forecast step',ylabel='Original units')
                ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=len(labels))
        fig.suptitle('Fixed windows, seed2024: first 96 steps of each horizon; no visual selection')
        fig.tight_layout(rect=(0,.035,1,.96))
        fig.savefig(root/f'local_detail_H{h}.png',dpi=160);fig.savefig(root/f'local_detail_H{h}.pdf');plt.close(fig)
    # Algebraic branch decomposition, not a causal removal/retraining experiment.
    import torch
    residual_figures=[]
    for h in HORIZONS:
        rec=next(r for r in tests if r['candidate']==selection['recommended'] and r['horizon']==h and r['seed']==2024)
        run=Path(rec['source_dir']);auxfile=run/'figures/intermediate_steps.npz'
        with np.load(auxfile) as aux:
            if 'median_skip' not in aux.files:continue
            delta=aux['median_skip'][0]
        with np.load(run/'figures/forecast_samples.npz') as f:
            q=f['quantiles'][0];truth=f['target'][0];names=[str(x) for x in f['feature_columns']]
        cfg=json.loads((run/'resolved_config.json').read_text())
        delta=delta*np.array(cfg['scaler']['scale'])[None,:]
        cp=torch.load(run/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        weights=cp['model_state']['skip_projection.weight'].numpy()
        gate=float(.5*torch.tanh(cp['model_state']['skip_gate']))
        fig,axes=plt.subplots(2,2,figsize=(14,7))
        ch=names.index('OT') if 'OT' in names else 0;n=min(h,96);t=np.arange(n)
        axes[0,0].plot(t,truth[:n,ch],color='black',label='Observed')
        axes[0,0].plot(t,q[:n,ch,3]-delta[:n,ch],label='Main-branch median')
        axes[0,0].plot(t,q[:n,ch,3],label='Final median')
        axes[0,0].set(title=f'H{h}, {names[ch]}, first fixed origin',ylabel='Original units');axes[0,0].legend(fontsize=8)
        axes[0,1].plot(t,delta[:n,ch]);axes[0,1].axhline(0,ls='--',color='gray')
        axes[0,1].set(title=f'Learned common quantile shift; gate={gate:.4f}',ylabel='Original units',xlabel='Forecast step')
        image=axes[1,0].imshow(weights,aspect='auto',origin='lower',cmap='coolwarm')
        fig.colorbar(image,ax=axes[1,0]);axes[1,0].set(title='Learned raw-history projection',xlabel='History index',ylabel='Forecast step')
        axes[1,1].plot(t,np.diff(truth[:n,ch],prepend=truth[0,ch]),color='black',label='Observed change')
        axes[1,1].plot(t,np.diff(q[:n,ch,3],prepend=q[0,ch,3]),label='Predicted change')
        axes[1,1].set(title='Local first differences',xlabel='Forecast step',ylabel='Original units');axes[1,1].legend(fontsize=8)
        fig.suptitle('Algebraic decomposition of a jointly trained model; not causal attribution')
        save(fig,root,f'residual_decomposition_H{h}');residual_figures.append(h)
    # All validation observations remain available, including unsuccessful candidates.
    screen=json.loads((root/'screen_records.json').read_text())
    flatten(screen).to_csv(root/'screen_metrics.csv',index=False)
    hist=json.loads((root/'history_selection.json').read_text())
    audit=dict(history=selection['history'],horizons=list(HORIZONS),confirmation_records=len(confirmed),test_records=len(tests),
               source_verified=True,artifacts_verified=True,three_seed_groups_complete=True,
               test_used_for_selection=False,selection_sha256=sha256(root/'selection.json'),scope=selection['scope'])
    audit['analysis_scripts_sha256']={p:sha256(p) for p in ['scripts/run_etth1_detail_search.py','scripts/freeze_etth1_horizon_choices.py','scripts/evaluate_etth1_detail_search.py','scripts/report_etth1_detail_search.py']}
    atomic_json(root/'final_audit.json',audit)
    show=testagg[['candidate','history','horizon','pinball_mean','pinball_std','delta_mae_mean','jump_mae_mean','PICP90_mean','PINAW90_mean','NormalizedWinkler90_mean']].copy()
    show['PICP90_mean']*=100
    html_text='''<!doctype html><meta charset="utf-8"><title>ETTh1结构与参数搜索</title>
<style>body{max-width:1600px;margin:25px auto;padding:20px;font:16px sans-serif;line-height:1.65;background:#f7f9fb}table{border-collapse:collapse;background:white;font-size:13px}td,th{border:1px solid #ccd;padding:7px}img{max-width:100%;height:auto}pre{white-space:pre-wrap}</style>
<h1>ETTh1：结构与参数搜索</h1><p>统一历史窗口：'''+str(selection['history'])+'''；预测96/192/336/720步；三种子2024/2025/2026。先验证选择历史窗口，再固定窗口筛选结构，最后三种子确认；测试在配置冻结后执行。</p>
<p>本结果是预先列出的有限空间与预算内最优，不是全局最优证明。短预算筛选可能遗漏慢收敛候选；多次使用验证集会产生选择乐观性；局部误差不能要求恢复不可预测噪声。</p><p>精度配置：'''+html.escape(selection['accuracy'])+'；细节配置：'+html.escape(selection['detail'])+'；细节约束满足：'+str(selection['detail_constraints_satisfied'])+'''。</p>
<p><a href="index.html">全部预测区间曲线</a> · <a href="test_aggregates.csv">测试均值/样本SD</a> · <a href="screen_metrics.csv">全部筛选记录</a> · <a href="selection.json">冻结选择</a> · <a href="learning_curves.pdf">训练曲线PDF</a></p>
<h2>历史长度预实验（仅验证）</h2>'''+format_table(pd.DataFrame(hist['ranking']))+'<h2>三种子验证排名</h2>'+format_table(ranking)+'<h2>最终测试：均值与样本标准差</h2>'+format_table(show)+'''<p>PICP90以百分数表示。差分MAE和大变化MAE在训练标准化单位下计算，越低越好；差分比较不含预测起点前的边界。大变化阈值仅由训练前缀拟合。覆盖率需结合宽度读取。</p>
<img src="metric_comparison.png" alt="测试损失与局部误差"><h2>训练与验证曲线</h2><img src="learning_curves.png" alt="训练验证曲线">'''
    for h in HORIZONS:html_text+=f'<h2>H{h}局部细节：固定三个窗口</h2><img src="local_detail_H{h}.png" alt="H{h}局部对比">'
    if horizon_choice:
        options=[dict(horizon=int(h),accuracy=v['accuracy'],detail=v['detail'],history=selection['history']) for h,v in horizon_choice['choices'].items()]
        html_text+='<h2>按预测长度的验证优选（测试前冻结，历史窗口统一）</h2>'+format_table(pd.DataFrame(options))
        for role in ('accuracy','detail'):
            summary=[]
            for h in HORIZONS:
                candidate=horizon_choice['choices'][str(h)][role]
                row=testagg[(testagg.candidate==candidate)&(testagg.horizon==h)].iloc[0]
                ref=testagg[(testagg.candidate=='reference')&(testagg.horizon==h)].iloc[0]
                summary.append(dict(horizon=h,candidate=candidate,history=selection['history'],
                    reference_pinball=ref.pinball_mean,pinball=row.pinball_mean,pinball_std=row.pinball_std,
                    pinball_improvement_percent=100*(1-row.pinball_mean/ref.pinball_mean),
                    delta_mae_improvement_percent=100*(1-row.delta_mae_mean/ref.delta_mae_mean),
                    jump_mae_improvement_percent=100*(1-row.jump_mae_mean/ref.jump_mae_mean),
                    PICP90_percent=100*row.PICP90_mean))
            summary=pd.DataFrame(summary);summary.to_csv(root/f'selected_{role}_summary.csv',index=False)
            html_text+='<h3>'+role+'（各H配置不同，不是同一结构的拼接排名）</h3>'+format_table(summary)

    if residual_figures:
        html_text+='<h2>残差支路与主分支分解</h2><p>以下为同一训练模型的代数分解，不是重新训练消融，也不等价于因果贡献证明。</p>'
        for h in residual_figures:html_text+=f'<img src="residual_decomposition_H{h}.png" alt="H{h}残差分解">'
    (root/'report.html').write_text(html_text)
    print(show.to_string(index=False));print('Recommended:',selection['recommended'],'history:',selection['history'])


if __name__=='__main__':main()
