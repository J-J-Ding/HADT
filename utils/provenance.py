"""Content fingerprints and atomic JSON for new experiment records."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)


def source_manifest():
    paths=[]
    for root in ('models','layers','utils','engine','data_provider','experiments'):
        paths.extend(Path(root).rglob('*.py'))
    paths.extend([Path('main.py'),Path('third_party/BASELINE_SOURCES.json')])
    return {str(p):sha256(p) for p in sorted(paths) if p.is_file()}


def identity(config, dataset, horizon, seed):
    data=Path(config['data']['root_path'])/config['data']['files'][dataset]
    source=source_manifest()
    payload=dict(config=config,dataset=dataset,horizon=int(horizon),seed=int(seed),
                 data_sha256=sha256(data),source=source)
    signature=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    return signature,payload
