"""Historical all-token decoder control; not an external baseline."""
from torch import nn
from layers.hadt_nq.dense_horizon import DenseHorizonAttention


class DenseNQ(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        # Adopt existing modules at the root to preserve historical state keys.
        self.__dict__.update(backbone.__dict__)
        c=config['model'].get('params',config['model'])
        decoder=DenseHorizonAttention(d_model=self.d_model,n_heads=self.decoder.n_heads,
            n_points=self.n_patches,max_horizon=self.decoder.horizon_embedding.num_embeddings,max_offset=self.decoder.max_offset,
            horizon_chunk_size=self.decoder.horizon_chunk_size,dropout=c.get('dropout',0.1),
            gradient_checkpointing=self.decoder.gradient_checkpointing)
        decoder.load_state_dict({k:v for k,v in self.decoder.state_dict().items() if not k.startswith('offset_projection.')})
        self.decoder=decoder
        self.n_points=self.n_patches

    def forward(self, *args, **kwargs):
        from models.hadt_nq import HADTNQ
        return HADTNQ.forward(self, *args, **kwargs)
