"""Explicit supported native long-term forecasting configurations."""
from types import SimpleNamespace

COMMON = dict(d_model=128, e_layers=2, d_ff=256, dropout=0.1)
DEFAULTS = {
    'dlinear': dict(individual=False),
    'patchtst': dict(COMMON, n_heads=8, fc_dropout=0.1, head_dropout=0.0,
                    individual=False, patch_len=16, stride=8, padding_patch='end',
                    revin=True, affine=True, subtract_last=False,
                    decomposition=False, kernel_size=25),
    'itransformer': dict(COMMON, n_heads=8, activation='gelu', output_attention=False,
                        use_norm=True, embed='timeF', freq='h', factor=1,
                        class_strategy='projection'),
    'timemixer': dict(COMMON, channel_independence=1, down_sampling_layers=2,
                     down_sampling_window=2, down_sampling_method='avg',
                     decomp_method='moving_avg', moving_avg=25, top_k=5,
                     use_norm=1, use_future_temporal_feature=False, embed='timeF', freq='h'),
}


def native_config(config, pred_len, n_vars):
    m = config['model']; name = m['name']
    if name not in DEFAULTS:
        raise ValueError(f'Unknown external model: {name}')
    supplied = m.get('params', {})
    unknown = set(supplied) - set(DEFAULTS[name])
    if unknown:
        raise ValueError(f'Unsupported {name} params: {sorted(unknown)}')
    p = dict(DEFAULTS[name], **supplied)
    p.update(seq_len=int(config['data']['seq_len']), pred_len=int(pred_len),
             enc_in=int(n_vars), c_out=int(n_vars), task_name='long_term_forecast', label_len=0)
    for key in ['d_model','e_layers','d_ff','n_heads','patch_len','stride','kernel_size','moving_avg','down_sampling_window']:
        if key in p and (not isinstance(p[key], int) or isinstance(p[key], bool) or p[key] <= 0):
            raise ValueError(f'{name}.{key} must be a positive integer')
    if p['seq_len'] <= 0 or p['pred_len'] <= 0 or p['enc_in'] <= 0:
        raise ValueError('Sequence lengths and number of variables must be positive')
    if 'n_heads' in p and p['d_model'] % p['n_heads']:
        raise ValueError('d_model must be divisible by n_heads')
    if not 0 <= p.get('dropout', 0) < 1:
        raise ValueError('dropout must lie in [0,1)')
    if name == 'patchtst':
        if p['patch_len'] > p['seq_len'] or p['padding_patch'] not in (None, 'end'):
            raise ValueError('Invalid PatchTST patch/padding configuration')
        if p['kernel_size'] % 2 != 1:
            raise ValueError('PatchTST decomposition requires odd kernel_size')
    if name == 'timemixer':
        if p['decomp_method'] != 'moving_avg' or p['down_sampling_method'] not in ('avg','max'):
            raise ValueError('TimeMixer first release supports moving_avg and avg/max downsampling only')
        if p['use_future_temporal_feature'] or p['channel_independence'] not in (0,1):
            raise ValueError('No future temporal features; channel_independence must be 0 or 1')
        count=p['down_sampling_layers']; window=p['down_sampling_window']
        if not isinstance(count,int) or count < 1 or window < 2 or p['seq_len'] % (window ** count):
            raise ValueError('TimeMixer requires >=1 scales, window>=2, and exactly divisible downsampled lengths')
        if p['moving_avg'] % 2 != 1:
            raise ValueError('TimeMixer requires odd moving_avg')
    if p.get('activation', 'gelu') not in ('relu','gelu'):
        raise ValueError('activation must be relu or gelu')
    return SimpleNamespace(**p)
