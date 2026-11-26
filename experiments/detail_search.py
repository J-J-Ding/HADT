"""Prespecified ETTh1 search space and validation-only selection rules."""
import copy
import json
from pathlib import Path
import numpy as np

HORIZONS=(96,192,336,720)
SEEDS=(2024,2025,2026)


def candidates():
    base=dict(encoder_kind='mlp',d_model=64,d_ff=128,e_layers=3,patch_len=16,stride=8,
              dropout=.1,n_points=8,max_offset=1.5,grid_power=1.,median_skip='none',
              skip_dropout=0.,local_kernel=3,learning_rate=3e-4,weight_decay=1e-5,difference_weight=0.)
    changes=[('reference',{}),
        ('mlp1_regular',dict(e_layers=1,dropout=.2,weight_decay=1e-3)),
        ('mlp2_regular',dict(e_layers=2,dropout=.2,weight_decay=1e-3)),
        ('mlp32',dict(d_model=32,d_ff=64,e_layers=2,dropout=.2,weight_decay=1e-3)),
        ('mlp96',dict(d_model=96,d_ff=192,e_layers=2,dropout=.2,weight_decay=1e-3)),
        ('mlp_slow',dict(learning_rate=1e-4,dropout=.2,weight_decay=1e-3)),
        ('mlp_fast',dict(learning_rate=1e-3,dropout=.2,weight_decay=1e-3)),
        ('fine_patch',dict(patch_len=8,stride=4,e_layers=2,dropout=.2,weight_decay=1e-3)),
        ('local_patch',dict(encoder_kind='local',e_layers=2,dropout=.2,weight_decay=1e-3)),
        ('local_fine',dict(encoder_kind='local',patch_len=8,stride=4,e_layers=2,dropout=.2,weight_decay=1e-3)),
        ('dilated_local',dict(encoder_kind='dilated',e_layers=2,local_kernel=3,dropout=.2,weight_decay=1e-3)),
        ('local_kernel7',dict(encoder_kind='local',e_layers=2,local_kernel=7,dropout=.2,weight_decay=1e-3)),
        ('more_samples',dict(n_points=16,max_offset=3.,e_layers=2,dropout=.2,weight_decay=1e-3)),
        ('recent_grid',dict(grid_power=2.,max_offset=3.,e_layers=2,dropout=.2,weight_decay=1e-3)),
        ('linear_skip',dict(median_skip='linear',e_layers=2,dropout=.2,weight_decay=1e-3,skip_dropout=.1)),
        ('seasonal_skip',dict(median_skip='seasonal',e_layers=2,dropout=.2,weight_decay=1e-3,skip_dropout=.1)),
        ('detail_loss',dict(difference_weight=.03,e_layers=2,dropout=.2,weight_decay=1e-3)),
        ('transformer1',dict(encoder_kind='transformer',e_layers=1,dropout=.2,weight_decay=1e-3))]
    return [dict(base,**delta,id=name) for name,delta in changes]


def configuration(candidate,history,epochs,stage):
    config=copy.deepcopy(json.loads(Path('configs/selected_architecture.json').read_text())['config'])
    config.pop('paper',None);config.pop('ablation',None)
    config['data']['seq_len']=int(history);config['data']['datasets']=['ETTh1']
    keys=set(candidate)-{'id','learning_rate','weight_decay','difference_weight'}
    params={k:candidate[k] for k in keys}
    params.update(n_heads=8,max_horizon=720,horizon_chunk_size=96,gradient_checkpointing=False,min_quantile_gap=1e-4)
    n=(history-params['patch_len'])//params['stride']+1
    params['n_points']=min(n,params['n_points'])
    config['model']=dict(name='hadt_detail',variant=candidate['id'],output_mode='quantile',quantiles=[.05,.1,.25,.5,.75,.9,.95],params=params)
    config['training'].update(batch_size=32,epochs=epochs,early_stopping_patience=7,
        learning_rate=candidate['learning_rate'],weight_decay=candidate['weight_decay'],channel_policy='full',difference_weight=candidate['difference_weight'])
    config['evaluation'].update(batch_size=32,channel_policy='full')
    config['experiment'].update(run_name='etth1_detail_search',horizons=list(HORIZONS),seeds=list(SEEDS),scope=stage)
    config['runtime'].update(device='cuda:0',torch_threads=3)
    config['visualization']=dict(enabled=False,max_channels=7)
    return config


def scores(groups,baseline):
    """Equal-horizon relative metrics; no test scores accepted here."""
    rows=[]
    for name,records in groups.items():
        values={}
        for metric in ('pinball','delta_mae','jump_mae'):
            ratios=[]
            for h in HORIZONS:
                current=[r['validation'][metric] for r in records if r['horizon']==h]
                base=[r['validation'][metric] for r in baseline if r['horizon']==h]
                if not current or not base:raise ValueError('Incomplete horizon group')
                ratios.append(np.mean(current)/max(np.mean(base),1e-12))
            values[metric+'_ratio']=float(np.mean(ratios))
        values['score']=.7*values['pinball_ratio']+.2*values['delta_mae_ratio']+.1*values['jump_mae_ratio']
        values['mean_pinball']=float(np.mean([r['validation']['pinball'] for r in records]))
        values['mean_coverage90']=float(np.mean([r['validation']['PICP90'] for r in records]))
        rows.append(dict(candidate=name,**values))
    return sorted(rows,key=lambda r:(r['score'],r['candidate']))
