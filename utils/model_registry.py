"""Explicit model construction without dependencies on experiment scripts."""
import inspect
from torch import nn
from utils.baseline_config import native_config
from utils.losses import DEFAULT_QUANTILES

EXTERNAL_MODELS = ('patchtst','timemixer','dlinear','itransformer')
MODEL_NAMES = ('hadt_nq','hadt_detail') + EXTERNAL_MODELS + ('linear_nq','patch_nq','dense_nq','fixed_nq')


def build_hadt(config, pred_len):
    from models.hadt_nq import HADTNQ
    from layers.hadt_nq.residual_encoder import ResidualTokenBlock
    c=config['model']; p=dict(c.get('params',c))
    allowed=set(inspect.signature(HADTNQ).parameters)-{'seq_len','pred_len'}
    m=HADTNQ(seq_len=int(config['data']['seq_len']),pred_len=int(pred_len),
             **{k:v for k,v in p.items() if k in allowed})
    kind=p.get('encoder_kind','transformer')
    if kind=='mlp':
        m.encoder=nn.Sequential(*[ResidualTokenBlock(m.d_model,p.get('d_ff',256),p.get('dropout',0.1)) for _ in range(p.get('e_layers',3))],nn.LayerNorm(m.d_model))
    elif kind!='transformer':
        raise ValueError('HADT encoder_kind must be transformer or mlp')
    m.ablation=config.get('ablation')
    m.model_name='hadt_nq';m.output_mode='quantile'
    return m


def build_model(config, pred_len, n_vars):
    c=config['model'];name=c.get('name','hadt_nq')
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model: {name}; choose {MODEL_NAMES}')
    if list(c.get('quantiles',DEFAULT_QUANTILES)) != list(DEFAULT_QUANTILES):
        raise ValueError('The frozen seven quantiles are required')
    if name == 'hadt_detail':
        from models.hadt_detail import DetailHADTNQ
        from models.hadt_nq import HADTNQ
        params=dict(c.get('params',{}))
        allowed=(set(inspect.signature(HADTNQ).parameters)|set(inspect.signature(DetailHADTNQ).parameters))-{'kwargs','seq_len','pred_len'}
        if set(params)-allowed:raise ValueError('Unknown detail model params')
        model=DetailHADTNQ(seq_len=config['data']['seq_len'],pred_len=pred_len,**params)
        model.native_config=params
    elif name in EXTERNAL_MODELS:
        from importlib import import_module
        classes=dict(patchtst='PatchTSTModel',timemixer='TimeMixerModel',dlinear='DLinearModel',itransformer='ITransformerModel')
        native=native_config(config,pred_len,n_vars)
        model=getattr(import_module('models.'+name),classes[name])(native,c.get('output_mode','quantile'))
        model.native_config=vars(native)
    else:
        if c.get('output_mode','quantile')!='quantile':
            raise ValueError('Historical HADT/internal controls support quantile mode only')
        base=build_hadt(config,pred_len)
        if name=='hadt_nq':model=base
        elif name=='linear_nq':
            from models.linear_nq import LinearNQ
            model=LinearNQ(base)
        elif name=='patch_nq':
            from models.patch_nq import PatchNQ
            model=PatchNQ(base)
        elif name=='dense_nq':
            from models.dense_nq import DenseNQ
            model=DenseNQ(base,config)
        else:
            from models.fixed_nq import FixedNQ
            model=FixedNQ(base)
        model.model_name=name;model.output_mode='quantile'
    model.variant=c.get('variant',c.get('output_mode','quantile'))
    return model
