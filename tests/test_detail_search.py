import unittest
import torch
from models.hadt_detail import DetailHADTNQ
from layers.hadt_nq.local_detail import LocalResidualBlock,RecentGridAttention
from utils.detail_objective import detail_objective
from utils.losses import multi_quantile_pinball_loss


class DetailSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_variants_shape_order_and_gradients(self):
        for encoder in ('mlp','local','dilated','transformer'):
            for skip in ('none','linear','seasonal'):
                m=DetailHADTNQ(seq_len=32,pred_len=8,patch_len=8,stride=4,d_model=16,n_heads=2,
                    e_layers=1,d_ff=32,dropout=0,encoder_kind=encoder,median_skip=skip,grid_power=2)
                x=torch.randn(2,32,5,requires_grad=True)
                q=m(x);self.assertEqual(q.shape,(2,8,5,7));self.assertTrue((q[...,1:]>=q[...,:-1]).all())
                loss=detail_objective(q,torch.randn(2,8,5),m.quantiles,.03);loss.backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(m.decoder.offset_projection.weight.grad.abs().sum(),0)
                if skip!='none':self.assertIsNotNone(m.skip_projection.weight.grad)

    def test_recent_grid_positions_and_attention(self):
        d=RecentGridAttention(grid_power=2,d_model=16,n_heads=2,n_points=4,dropout=0)
        x=torch.randn(2,9,16)
        _,offset,pos,att=d(x,5,return_attention=True)
        expected=torch.tensor([0,40/9,64/9,8.])
        torch.testing.assert_close(pos[0,0,0],expected,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(att.sum(-1),torch.ones_like(att.sum(-1)))
        self.assertTrue((offset==0).all())

    def test_local_block_mixes_neighbouring_positions(self):
        m=LocalResidualBlock(8,16,0).eval();x=torch.zeros(1,7,8)
        changed=x.clone();changed[:,3,0]=1
        with torch.no_grad():diff=(m(changed)-m(x)).abs().sum(-1)
        self.assertGreater(float(diff[0,2]),0);self.assertGreater(float(diff[0,4]),0)
        self.assertEqual(float(diff[0,0]),0)

    def test_zero_detail_weight_is_original_loss(self):
        p=torch.randn(2,8,5,7);y=torch.randn(2,8,5);q=[.05,.1,.25,.5,.75,.9,.95]
        self.assertTrue(torch.equal(detail_objective(p,y,q,0),multi_quantile_pinball_loss(p,y,q)))

    def test_jump_threshold_uses_training_prefix_only(self):
        import numpy as np
        from types import SimpleNamespace
        from utils.forecast_detail_metrics import training_jump_thresholds
        a=np.stack([np.arange(20),2*np.arange(20)],axis=1).astype(float)
        b=np.concatenate([a,np.ones((10,2))*1e9])
        bundle=SimpleNamespace(train_loader=SimpleNamespace(dataset=SimpleNamespace(values=b)),split_bounds={'train':(0,20)})
        np.testing.assert_allclose(training_jump_thresholds(bundle),[1,2])

    def test_detail_metrics_on_known_forecast(self):
        import numpy as np
        from types import SimpleNamespace
        from utils.forecast_detail_metrics import evaluate_details
        class Known(torch.nn.Module):
            quantiles=(.05,.1,.25,.5,.75,.9,.95)
            def forward(self,x):
                median=x[:,:4,:]
                return median.unsqueeze(-1)+torch.tensor([-3,-2,-1,0,1,2,3.])
        x=torch.arange(8.).reshape(1,4,2);scaler=SimpleNamespace(scale_=np.ones(2),data_range_=np.ones(2)*6)
        r=evaluate_details(Known(),[(x,x)],torch.device('cpu'),np.ones(2),scaler)
        self.assertEqual(r['delta_mae'],0);self.assertEqual(r['jump_mae'],0)
        self.assertEqual(r['roughness_ratio'],1);self.assertEqual(r['PICP90'],1)
        self.assertAlmostEqual(r['PINAW90'],1)
