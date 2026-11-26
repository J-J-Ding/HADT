"""Contracts that catch baseline distortion, scalar-head bypass and context loss."""
import copy
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils.config import load_config, validate_config
from utils.model_registry import build_model
from utils.quantile_heads import MedianAnchoredNonCrossingHead
from utils.losses import forecasting_loss
from utils.channel_policy import resolve_policy, channel_slices
from engine.checkpoint import save_checkpoint, load_checkpoint

NAMES=('dlinear','patchtst','itransformer','timemixer')


def config(name, point=False):
    c=load_config(f'configs/baselines/{name}.yaml')
    c['data']['seq_len']=32;c['experiment']['horizons']=[8]
    p=c['model']['params']
    for key, val in dict(d_model=16,d_ff=32,e_layers=1,n_heads=2,patch_len=8,stride=4,moving_avg=5).items():
        if key in p:p[key]=val
    c['model']['output_mode']='point' if point else 'quantile'
    c['runtime']['device']='cpu'
    return c


class ExternalBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_head_preserved_scalar_adapter_and_gradients(self):
        for name in NAMES:
            with self.subTest(name=name):
                torch.manual_seed(42);m=build_model(config(name),8,5).eval()
                x=torch.randn(2,32,5,requires_grad=True);before=x.detach().clone()
                captured=[]
                hook=m.quantile_head.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
                q=m(x);hook.remove()
                p=m.point_model(x.clone(),None,None,None) if m.multi_argument else m.point_model(x.clone())
                if isinstance(p,tuple):p=p[0]
                torch.testing.assert_close(captured[0],p.permute(0,2,1).reshape(10,8,1),rtol=0,atol=0)
                self.assertIs(type(m.quantile_head),MedianAnchoredNonCrossingHead)
                self.assertEqual(m.quantile_head.projection.in_features,1)
                self.assertEqual(q.shape,(2,8,5,7))
                self.assertTrue((q[...,1:]>=q[...,:-1]).all())
                self.assertTrue(torch.equal(x.detach(),before))
                loss=forecasting_loss(q,torch.randn(2,8,5));loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(x.grad).all())
                for key in ['point_model','quantile_head']:
                    params=list(getattr(m,key).parameters())
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in params),key)
                point=build_model(config(name,True),8,5)
                self.assertEqual(set(point.point_model.state_dict()),set(m.point_model.state_dict()))
                for k,v in point.point_model.state_dict().items():self.assertEqual(v.shape,m.point_model.state_dict()[k].shape)

    def test_point_contract_and_unsupported_intervention(self):
        for name in NAMES:
            m=build_model(config(name,True),8,5).eval()
            x=torch.randn(2,32,5)
            y,aux=m(x,return_aux=True)
            self.assertEqual(y.shape,(2,8,5));self.assertEqual(aux,{})
            self.assertIsNone(m.quantile_head)
            self.assertTrue(torch.isfinite(forecasting_loss(y,torch.ones_like(y))))
            with self.assertRaises(ValueError):m(x,intervention='no_offsets')

    def test_context_policy_and_fixed_channel_heads(self):
        for name in NAMES:
            c=config(name);m=build_model(c,8,5)
            self.assertIsNone(resolve_policy(m,c)['train_sample_size'])
            if name!='dlinear':
                self.assertIsNone(resolve_policy(m,c)['evaluation_chunk_size'])
                c['training']['channel_policy']='sample'
                with self.assertRaises(ValueError):resolve_policy(m,c)
                c['training']['channel_policy']='auto';c['evaluation']['channel_policy']='chunk'
                with self.assertRaises(ValueError):resolve_policy(m,c)
                with self.assertRaises(ValueError):m(torch.randn(2,32,3))
        c=config('dlinear');c['model']['params']['individual']=True;m=build_model(c,8,5)
        c['evaluation']['channel_policy']='chunk'
        with self.assertRaises(ValueError):resolve_policy(m,c)

    def test_shared_dlinear_chunk_equivalence_and_gradient(self):
        m=build_model(config('dlinear'),8,5).eval();x=torch.randn(2,32,5,requires_grad=True)
        full=m(x);parts=torch.cat([m(x[:,:,s]) for s in channel_slices(m,5,2)],dim=2)
        torch.testing.assert_close(full,parts,atol=1e-7,rtol=1e-6)
        g1=torch.autograd.grad(full.sum(),x,retain_graph=True)[0]
        g2=torch.autograd.grad(parts.sum(),x)[0]
        torch.testing.assert_close(g1,g2,atol=1e-7,rtol=1e-6)

    def test_invalid_model_specific_config(self):
        for name in NAMES:
            c=config(name);c['model']['params']['not_used']=1
            with self.assertRaises(ValueError):validate_config(c)
        c=config('dlinear');c['model']['params']['patch_len']=16
        with self.assertRaises(ValueError):validate_config(c)
        c=config('timemixer');c['data']['seq_len']=33
        with self.assertRaises(ValueError):validate_config(c)

    def test_checkpoint_roundtrip_and_metadata_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in NAMES:
                c=config(name);m=build_model(c,8,5).eval();x=torch.randn(2,32,5)
                opt=torch.optim.AdamW(m.parameters());path=Path(tmp)/(name+'.pt')
                save_checkpoint(path,m,opt,1,.1,c)
                clone=build_model(c,8,5).eval();load_checkpoint(path,clone)
                torch.testing.assert_close(m(x),clone(x),atol=0,rtol=0)
                point=build_model(config(name,True),8,5)
                with self.assertRaises(ValueError):load_checkpoint(path,point)
                if name == 'itransformer':
                    different=config(name);different['model']['params']['use_norm']=False
                    with self.assertRaises(ValueError):load_checkpoint(path,build_model(different,8,5))

    def test_plot_selection_preserves_full_itransformer_context(self):
        from types import SimpleNamespace
        from engine.visualization import create_run_figures
        from data_provider.scaler import StandardScaler
        import pandas as pd
        m=build_model(config('itransformer'),8,5).eval()
        history=torch.randn(3,32,5);target=torch.randn(3,8,5)
        bundle=SimpleNamespace(test_loader=DataLoader(TensorDataset(history,target)),
            scaler=StandardScaler().fit(np.stack([np.zeros(5),np.ones(5)*2])),
            feature_columns=list('abcde'))
        seen=[];hook=m.register_forward_pre_hook(lambda module,args:seen.append(args[0].shape[-1]))
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(dict(epoch=[1],train_pinball=[.2],val_pinball=[.3])).to_csv(Path(tmp)/'training_history.csv',index=False)
            create_run_figures(m,bundle,torch.device('cpu'),tmp,dict(PICP50=.5,PICP80=.8,PICP90=.9),max_channels=2)
            data=np.load(Path(tmp)/'figures/forecast_samples.npz')
            self.assertEqual(seen,[5,5,5]);self.assertEqual(data['quantiles'].shape,(3,8,2,7))
            expected=m(history).detach().numpy()[:,:,[0,4]]+1
            np.testing.assert_allclose(data['quantiles'],expected,rtol=1e-6,atol=1e-6)
        hook.remove()

    def test_evaluation_keeps_full_context_and_point_has_no_intervals(self):
        from engine.evaluator import evaluate_test
        from data_provider.scaler import StandardScaler
        scaler=StandardScaler().fit(np.stack([np.zeros(5),np.ones(5)*2]))
        x=torch.randn(2,32,5);y=torch.randn(2,8,5)
        for point in (False,True):
            m=build_model(config('itransformer',point),8,5)
            with tempfile.TemporaryDirectory() as tmp:
                metrics=evaluate_test(m,[(x,y)],torch.device('cpu'),scaler,2,[.05,.1,.25,.5,.75,.9,.95],tmp)
                self.assertTrue('MSE_standardized' in metrics if point else 'PICP90' in metrics)
                self.assertEqual('PICP90' in metrics,not point)
                self.assertEqual((Path(tmp)/'horizon_metrics.csv').exists(),not point)

    def test_internal_models_registry(self):
        c=load_config('configs/default.yaml');c['data']['seq_len']=32;c['model'].update(patch_len=8,stride=4,d_model=16,d_ff=32,n_heads=2,e_layers=1)
        for name in ['hadt_nq','linear_nq','patch_nq','dense_nq','fixed_nq']:
            c['model']['name']=name;m=build_model(c,8,5).eval()
            q,aux=m(torch.randn(2,32,5),return_aux=True)
            self.assertEqual(q.shape,(2,8,5,7))

    def test_point_run_and_cache_conflict_protection(self):
        from experiments.runner import run_single_experiment
        from types import SimpleNamespace
        from data_provider.scaler import StandardScaler
        x=torch.randn(3,32,5);y=torch.randn(3,8,5)
        bundle=SimpleNamespace(train_loader=DataLoader(TensorDataset(x,y),batch_size=2),
            val_loader=DataLoader(TensorDataset(x,y),batch_size=2),
            test_loader=DataLoader(TensorDataset(x,y),batch_size=2),
            scaler=StandardScaler().fit(np.stack([np.zeros(5),np.ones(5)*2])),
            feature_columns=list('abcde'),n_vars=5,n_rows=3,split_bounds={})
        with tempfile.TemporaryDirectory() as tmp:
            c=config('dlinear',True);c['training']['epochs']=1
            c['experiment'].update(output_dir=tmp,run_name='integration_test')
            c['visualization'].update(enabled=True,max_channels=1)
            with patch('experiments.runner.build_data_bundle',return_value=bundle):
                result=run_single_experiment(c,'ETTh1',8,2024)
                self.assertIn('best_val_mse',result);self.assertNotIn('PICP90',result)
                marker=next(Path(tmp).rglob('complete.json'));run=marker.parent
                self.assertTrue((run/'figures/predictions/overview.png').exists())
                again=run_single_experiment(c,'ETTh1',8,2024)
                self.assertEqual(result,again)
                bad=copy.deepcopy(c);bad['training']['learning_rate']*=2
                with self.assertRaises(ValueError):run_single_experiment(bad,'ETTh1',8,2024)
                (run/'metrics.json').write_text('{}')
                with self.assertRaises(ValueError):run_single_experiment(c,'ETTh1',8,2024)

    def test_high_variable_count_keeps_all_tokens(self):
        for name in ('itransformer','timemixer'):
            m=build_model(config(name),8,73).eval()
            with torch.no_grad():q=m(torch.randn(1,32,73))
            self.assertEqual(q.shape,(1,8,73,7))
            self.assertEqual(list(channel_slices(m,73,8)),[slice(0,73)])
