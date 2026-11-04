"""Independent artifact checks: measured values, data partitions, selected weights, and plot inputs."""
import hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from data_provider.csv_dataset import read_multivariate_csv
from data_provider.scaler import StandardScaler
from scripts.export_paper_evidence import load_records,aggregate_complete_groups,sample_path,PAPER,METRICS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    records=load_records()
    sensitivity_path=Path('outputs/paper_sensitivity/records.json')
    sensitivity=json.loads(sensitivity_path.read_text()) if sensitivity_path.exists() else []
    all_records=records+sensitivity
    datasets={};verified=[]
    for r in all_records:
        source=Path(r['source_dir'])
        cp=torch.load(source/'best_checkpoint.pt',map_location='cpu',weights_only=False)
        resolved=json.loads((source/'resolved_config.json').read_text())
        cfg=resolved['config'];dataset=r['dataset'];h=int(r['horizon'])
        if dataset not in datasets:
            path=Path(cfg['data']['root_path'])/cfg['data']['files'][dataset]
            raw,names=read_multivariate_csv(path,cfg['data']['time_column'])
            ratio='ETT' if dataset.startswith('ETT') else 'other'
            train_ratio,val_ratio,_=cfg['data']['split_ratios'][ratio]
            train_end=int(len(raw)*train_ratio);val_end=train_end+int(len(raw)*val_ratio)
            scaler=StandardScaler().fit(raw[:train_end])
            frame=pd.read_csv(path,usecols=[cfg['data']['time_column']])
            dates=pd.to_datetime(frame.iloc[:,0])
            assert dates.is_monotonic_increasing and not dates.duplicated().any()
            datasets[dataset]=dict(raw=raw,names=names,scaler=scaler,sha256=sha(path),path=str(path),
                        rows=len(raw),variables=len(names),train_end=train_end,val_end=val_end,
                        date_start=str(dates.iloc[0]),date_end=str(dates.iloc[-1]))
        ds=datasets[dataset]
        assert r['data_sha256']==ds['sha256']
        assert resolved['feature_columns']==ds['names']
        assert resolved['n_rows']==ds['rows'] and resolved['n_vars']==ds['variables']
        assert list(resolved['split_bounds']['train'])==[0,ds['train_end']]
        assert list(resolved['split_bounds']['val'])==[ds['train_end'],ds['val_end']]
        assert list(resolved['split_bounds']['test'])==[ds['val_end'],ds['rows']]
        for key,value in ds['scaler'].state_dict().items():
            np.testing.assert_allclose(cp['scaler'][key],value,rtol=0,atol=1e-12)
        hist=pd.read_csv(source/'training_history.csv')
        assert cp['best_val_loss']<=float(hist.val_pinball.min())+1e-8
        assert np.isclose(hist.loc[hist.epoch==cp['epoch'],'val_pinball'].iloc[0],cp['best_val_loss'],rtol=0,atol=1e-12)
        assert np.isclose(r['best_val_pinball'],cp['best_val_loss'],rtol=0,atol=1e-12)
        raw_metrics=json.loads((source/'metrics.json').read_text())
        for metric in METRICS:
            if metric in raw_metrics and metric in r:
                assert np.isclose(raw_metrics[metric],r[metric],rtol=0,atol=1e-12)
        assert np.isfinite(r['QuantileLoss_standardized']) and r['QuantileLoss_standardized']>=0
        for nominal in [50,80,90]:assert 0<=r[f'PICP{nominal}']<=1
        if r['method']!='independent':assert r['CrossingRate']==0
        data=np.load(sample_path(r));names=list(data['feature_columns'])
        channels=data['channel_indices'] if 'channel_indices' in data else np.array([ds['names'].index(n) for n in names])
        np.testing.assert_array_equal(np.array(ds['names'])[channels],data['feature_columns'])
        n_windows=ds['rows']-ds['val_end']-h+1
        np.testing.assert_array_equal(data['sample_indices'],[0,n_windows//2,n_windows-1])
        assert data['quantiles'].shape==(3,h,len(channels),7)
        assert np.isfinite(data['quantiles']).all()
        if r['method']!='independent':assert (np.diff(data['quantiles'],axis=-1)>=-1e-10).all()
        for i,window in enumerate(data['sample_indices']):
            start=ds['val_end']+int(window)
            np.testing.assert_allclose(data['target'][i],ds['raw'][start:start+h,channels],rtol=1e-5,atol=1e-5)
            np.testing.assert_allclose(data['history'][i],ds['raw'][start-336:start,channels],rtol=1e-5,atol=1e-5)
        aux_file=sample_path(r,'intermediate_steps.npz')
        if r['method']=='hadt' and aux_file.exists():
            aux=np.load(aux_file)
            np.testing.assert_allclose(aux['attention'].sum(-1),1.,rtol=1e-5,atol=1e-5)
            assert aux['positions'].min()>=0 and aux['positions'].max()<=40
        verified.append(dict(dataset=dataset,method=r['method'],horizon=h,seed=r['seed'],
                      sensitivity_K=r.get('sensitivity_K'),source_dir=str(source),best_epoch=cp['epoch'],
                      epochs=len(hist),test_windows=n_windows,test_elements=n_windows*h*ds['variables'],
                      metrics_checked=True,train_only_scaler_checked=True,plot_truth_checked=True))
        print('VERIFIED',dataset,r['method'],h,r['seed'],r.get('sensitivity_K',''),flush=True)
    expected=aggregate_complete_groups(records)
    actual=pd.read_csv(PAPER/'results/aggregate_metrics.csv')
    assert len(expected)==len(actual)
    for row in expected.to_dict('records'):
        match=actual[(actual.dataset==row['dataset'])&(actual.method==row['method'])&(actual.horizon==row['horizon'])].iloc[0]
        assert match['n']==3
        for k,v in row.items():
            if isinstance(v,(float,np.floating)) and np.isfinite(v):assert np.isclose(match[k],v,rtol=1e-12,atol=1e-12)
    summaries={k:{a:v for a,v in d.items() if a not in ('raw','names','scaler')} for k,d in datasets.items()}
    (PAPER/'results/datasets.json').write_text(json.dumps(summaries,indent=2))
    result=dict(status='passed',completed_records=len(verified),primary_aggregate_groups=len(actual),
                checks=verified,datasets=summaries,
                scope='Checks recorded metrics against source JSON, training-only scaling, checkpoint selection and saved plot truth. Does not rerun every full-test neural prediction.')
    (PAPER/'results/artifact_validation.json').write_text(json.dumps(result,indent=2))
    print('All artifact checks passed:',len(verified),'completed records;',len(actual),'complete three-seed groups.')


if __name__=='__main__':main()
