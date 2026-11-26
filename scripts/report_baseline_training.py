"""Inspect completed convergence experiments without selecting on test metrics."""
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
from utils.provenance import atomic_json,sha256,source_manifest


def make_report(root):
    table=pd.read_csv(root/'training_summary.csv')
    comparison=pd.read_csv(root/'quantile_comparison.csv')
    assert len(table)==8
    rows=[]
    for record in json.loads((root/'launch.json').read_text())['configs']:
        name=record['model']
        for mode in ('point','quantile'):
            run=root/name/mode/'ETTh1/H96/seed2024'
            complete=json.loads((run/'complete.json').read_text())
            for file,digest in complete['artifact_sha256'].items():
                assert sha256(run/file)==digest,(run,file)
            recorded=json.loads((run/'source_manifest.json').read_text())
            assert recorded['source']==source_manifest()
            resolved=json.loads((run/'resolved_config.json').read_text())
            assert resolved['visible_devices']==str(record['gpu'])
            hist=pd.read_csv(run/'training_history.csv')
            assert len(hist)<=30 and len(hist)>=1
            val='val_mse' if mode=='point' else 'val_pinball'
            row=table[(table.model==name)&(table['mode']==mode)].iloc[0]
            assert int(row.best_epoch)==int(hist.loc[hist[val].idxmin(),'epoch'])
            rows.append(dict(model=name,mode=mode,gpu=record['gpu'],epochs_run=len(hist),best_epoch=int(row.best_epoch),source_and_artifacts_verified=True))
    fig,axes=plt.subplots(2,4,figsize=(17,7),squeeze=False)
    for col,name in enumerate(('dlinear','patchtst','itransformer','timemixer')):
        for row,mode in enumerate(('point','quantile')):
            hist=pd.read_csv(root/name/mode/'ETTh1/H96/seed2024/training_history.csv')
            metric='mse' if mode=='point' else 'pinball'
            ax=axes[row,col]
            ax.plot(hist.epoch.to_numpy(),hist['train_'+metric].to_numpy(),label='Train')
            ax.plot(hist.epoch.to_numpy(),hist['val_'+metric].to_numpy(),label='Validation')
            best=hist.loc[hist['val_'+metric].idxmin(),'epoch']
            ax.axvline(best,ls='--',color='gray',lw=.8,label=f'Best {int(best)}')
            ax.set(title=f'{name} | {mode}',xlabel='Epoch',ylabel='Standardized '+metric)
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('ETTh1, L336 to H96, seed2024 | validation selection | 30-epoch cap')
    fig.tight_layout();fig.savefig(root/'convergence.png',dpi=150);fig.savefig(root/'convergence.pdf');plt.close(fig)
    qtable=comparison[['model','source','QuantileLoss_standardized','PICP90','PINAW90','NormalizedWinkler90','epochs_run','best_epoch']].copy()
    qtable['PICP90_percent']=qtable.PICP90*100
    qtable['PICP90_error_pp']=(qtable.PICP90-.9)*100
    improvements=[]
    for name in ('dlinear','patchtst','itransformer','timemixer'):
        path=Path('outputs/baseline_integration_v1_release')/name/'quantile/ETTh1/H96/seed2024/metrics.json'
        old=json.loads(path.read_text())
        new=comparison[comparison.model==name].iloc[0]
        improvements.append(dict(model=name,one_epoch_pinball=old['QuantileLoss_standardized'],trained_pinball=float(new.QuantileLoss_standardized),
                                 relative_change_percent=(float(new.QuantileLoss_standardized)/old['QuantileLoss_standardized']-1)*100))
    pd.DataFrame(improvements).to_csv(root/'one_epoch_vs_trained.csv',index=False)
    ptable=table[table['mode']=='point'][['model','MSE_standardized','MAE_standardized','epochs_run','best_epoch']]
    atomic_json(root/'verification.json',dict(scope='One dataset/horizon/seed; untuned configs, capped budget. Historical HADT references differ in architecture/batch/LR policy. No significance or SOTA claim.',runs=rows,passed=True))
    naive_file=root/'naive_point_checks.csv'
    naive_html=('<h2>简单点预测检查</h2>'+pd.read_csv(naive_file).to_html(index=False,float_format=lambda x:f'{x:.6f}')) if naive_file.exists() else ''
    text='''<!doctype html><meta charset="utf-8"><title>ETTh1基线训练结果</title>
<style>body{max-width:1400px;margin:30px auto;padding:20px;font:16px sans-serif;line-height:1.6;background:#f7f8fa}table{border-collapse:collapse;background:white;font-size:14px}td,th{padding:8px;border:1px solid #ccd}img{width:100%;height:auto}</style>
<h1>ETTh1基线收敛检查</h1><p>L=336，H=96，seed=2024。四张GPU分别训练一个模型；原生点预测和分位数适配各自从初始化开始训练，最多30轮，按验证损失早停和选模。</p>
<p>这是单种子、未做超参数搜索的收敛检查。HADT行是历史参考，训练批量、编码器与学习率等不完全相同；不能据此作严格公平排名或稳定性结论。</p>
<p><a href="index.html">浏览8次运行的预测曲线</a> · <a href="training_summary.csv">全部指标CSV</a> · <a href="quantile_comparison.csv">分位数比较CSV</a> · <a href="convergence.pdf">收敛图PDF</a></p>
<h2>原生点预测</h2>'''+ptable.to_html(index=False,float_format=lambda x:f'{x:.6f}')+naive_html+'<h2>分位数适配与历史参考</h2>'+qtable.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'<p>损失和区间评分越低越好；PICP90应接近90%，宽度必须结合覆盖率解读。</p><h2>与单轮试跑比较</h2>'+pd.DataFrame(improvements).to_html(index=False,float_format=lambda x:f'{x:.6f}')+'<h2>训练与验证曲线</h2><img src="convergence.png" alt="八次训练的损失曲线">'
    (root/'report.html').write_text(text)
    print(qtable.to_string(index=False));print(ptable.to_string(index=False));print('Report:',root/'report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/baseline_etth1_30ep_seed2024'))
    make_report(p.parse_args().root)
