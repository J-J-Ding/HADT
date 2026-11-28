"""Commit research sources/results and push the existing branch, without force."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.provenance import atomic_json
OUT=ROOT/'outputs/paper_hourly_monitor'
ALLOWED_ROOTS={'HADT_NQ_ICCRD2027_LaTeX','scripts','docs','specs','models','layers','utils','engine','data_provider','experiments','configs','third_party','tests'}
ALLOWED_SUFFIXES={'.py','.md','.tex','.bib','.cls','.bst','.csv','.json','.yaml','.yml','.sh','.txt','.pdf','.png','.svg'}


def git(*args,check=True):
    env=os.environ.copy();env['GIT_TERMINAL_PROMPT']='0'
    env.setdefault('GIT_SSH_COMMAND','ssh -o BatchMode=yes -o ConnectTimeout=15')
    return subprocess.run(['git',*args],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=120,check=check)


def eligible(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:return False
    if len(p.parts)==1:return name in {'.gitignore','README.md','README_CN.md','requirements.txt','main.py'}
    if p.parts[0] not in ALLOWED_ROOTS or p.suffix not in ALLOWED_SUFFIXES:return False
    if any(part in ('__pycache__','.venv','node_modules') for part in p.parts):return False
    return not (ROOT/p).exists() or (ROOT/p).stat().st_size<=20*1024*1024


def backup():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'git_backup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    branch=git('branch','--show-current').stdout.strip()
    policy=OUT/'git_policy.json'
    if policy.exists():
        p=json.loads(policy.read_text());assert branch==p['branch'],'Current branch changed; refusing an automatic cross-branch push'
    else:
        assert branch,'Detached HEAD is not eligible for automatic backup'
        p=dict(remote='origin',branch=branch,force_push=False);atomic_json(policy,p)
    already=[x for x in git('diff','--cached','--name-only','-z').stdout.split('\0') if x]
    assert all(eligible(x) for x in already),'Unmanaged staged files require review; left untouched'
    names=git('ls-files','--modified','--deleted','--others','--exclude-standard','-z').stdout.split('\0')
    selected=sorted({x for x in names if x and eligible(x)})
    if selected:git('add','--',*selected)
    changed=git('diff','--cached','--quiet',check=False).returncode
    if changed not in (0,1):raise RuntimeError('Cannot inspect staged changes')
    created=False
    if changed==1:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M CST')
        git('commit','-m',f'paper: experiment and manuscript backup {stamp}');created=True
    head=git('rev-parse','HEAD').stdout.strip()
    result=git('push',p['remote'],f'HEAD:refs/heads/{p["branch"]}',check=False)
    if result.returncode:
        report=dict(status='push_failed_local_commit_preserved',commit=head,branch=branch,remote=p['remote'],
            error=result.stderr[-2000:],time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),force_push=False)
        atomic_json(OUT/'git_backup.json',report);raise RuntimeError('Git push failed; local commit preserved: '+result.stderr[-1000:])
    remote=git('ls-remote',p['remote'],f'refs/heads/{branch}').stdout.split()
    assert remote and remote[0]==head,'Remote HEAD verification failed'
    report=dict(status='pushed_verified',commit=head,branch=branch,remote=p['remote'],new_commit=created,
        time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),force_push=False,selected_paths=len(selected))
    atomic_json(OUT/'git_backup.json',report)
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    try:backup()
    except BaseException as error:
        if not (OUT/'git_backup.json').exists() or json.loads((OUT/'git_backup.json').read_text()).get('status')=='pushed_verified':
            atomic_json(OUT/'git_backup.json',dict(status='backup_failed',error=str(error) or repr(error),time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()))
        raise
