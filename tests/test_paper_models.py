import unittest
import torch
from utils.config import load_config


class TestPaperModels(unittest.TestCase):
    def cfg(self):
        c=load_config('configs/etth1_gpu3.yaml')
        c['data']['seq_len']=32
        c['model'].update(patch_len=8,stride=4,d_model=16,n_heads=2,e_layers=1,d_ff=32,
                          dropout=0.,max_horizon=8,horizon_chunk_size=3,gradient_checkpointing=False)
        return c

    def test_controls_shape_order_and_channel_independence(self):
        from experiments.paper_controls import make_paper_model
        x=torch.randn(2,32,3)
        for method in ['hadt','linear','patch','dense','fixed','shared','independent','no_revin']:
            with self.subTest(method=method):
                m=make_paper_model(self.cfg(),8,method).eval()
                q=m(x)
                self.assertEqual(q.shape,(2,8,3,7))
                self.assertTrue(torch.isfinite(q).all())
                if method!='independent':self.assertTrue((q[...,1:]>=q[...,:-1]).all())
                torch.testing.assert_close(q[:,:,1:2],m(x[:,:,1:2]),rtol=1e-4,atol=1e-5)
                q.mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters()))

    def test_shared_query_has_no_horizon_distinction_at_inference(self):
        from experiments.paper_controls import make_paper_model
        m=make_paper_model(self.cfg(),8,'shared').eval()
        q=m(torch.randn(2,32,3))
        torch.testing.assert_close(q,q[:,:1].expand_as(q),rtol=1e-5,atol=1e-6)

    def test_dense_equals_zero_offset_all_token_sparse_attention(self):
        from layers.hadt_nq.dense_horizon import DenseHorizonAttention
        from layers.hadt_nq.horizon_deformable import HorizonDeformableAttention
        args=dict(d_model=16,n_heads=2,n_points=7,max_horizon=8,dropout=0.,horizon_chunk_size=3)
        sparse=HorizonDeformableAttention(**args).eval()
        dense=DenseHorizonAttention(**args).eval()
        dense.load_state_dict({k:v for k,v in sparse.state_dict().items() if not k.startswith('offset_projection.')})
        z=torch.randn(3,7,16,requires_grad=True)
        with torch.no_grad():
            sparse.offset_projection.weight.normal_(0.,.1)
        a=sparse(z,8,disable_offsets=True)[0]
        b=dense(z,8)[0]
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        ga=torch.autograd.grad(a.square().sum(),z)[0]
        gb=torch.autograd.grad(b.square().sum(),z)[0]
        torch.testing.assert_close(ga,gb,rtol=1e-4,atol=1e-5)
