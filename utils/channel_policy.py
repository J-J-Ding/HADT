"""Preserve variable context in every training, evaluation and plotting path."""


def resolve_policy(model, config):
    training=config['training']; evaluation=config['evaluation']
    train_policy=training.get('channel_policy','auto')
    eval_policy=evaluation.get('channel_policy','auto')
    if train_policy not in ('auto','full','sample') or eval_policy not in ('auto','full','chunk'):
        raise ValueError('Invalid channel_policy: training auto/full/sample; evaluation auto/full/chunk')
    subsets=getattr(model,'supports_channel_subsets',True)
    chunks=getattr(model,'supports_channel_chunking',True)
    if train_policy=='sample' and not subsets:
        raise ValueError('This model requires full variable context; sampling is prohibited')
    if eval_policy=='chunk' and not chunks:
        raise ValueError('This model requires full variable context; chunking is prohibited')
    sample=training.get('channel_sample_size',16)
    if train_policy=='full' or (train_policy=='auto' and not getattr(model,'default_channel_sampling',True)):
        sample=None
    if train_policy=='auto' and not subsets:
        sample=None
    chunk=evaluation.get('channel_chunk_size',8)
    if eval_policy=='full' or not chunks:
        chunk=None
    if sample is not None and int(sample) < 1 or chunk is not None and int(chunk) < 1:
        raise ValueError('Channel sizes must be positive or None')
    return dict(train_sample_size=sample, evaluation_chunk_size=chunk)


def channel_slices(model, channels, chunk_size):
    if not getattr(model,'supports_channel_chunking',True) or chunk_size is None:
        yield slice(0,channels)
        return
    if int(chunk_size)<1:
        raise ValueError('channel_chunk_size must be positive')
    for start in range(0,channels,int(chunk_size)):
        yield slice(start,min(start+int(chunk_size),channels))
