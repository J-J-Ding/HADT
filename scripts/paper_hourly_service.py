"""Detached hourly result publisher and automatic V4 handoff; zero model API calls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.provenance import atomic_json
OUT=ROOT/'outputs/paper_hourly_monitor'
REVIEW=ROOT/'outputs/reviewer_revision_20261128'
LEGACY=ROOT/'outputs/paper_revision_20261126/finalization_progress.json'


def read(path):
    try:return json.loads(Path(path).read_text()) if Path(path).exists() else {}
    except json.JSONDecodeError:return {}


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def active(script):
    matches=[];target=(ROOT/'scripts'/script).resolve()
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if p.stat().st_uid!=os.getuid():continue
            argv=p.read_bytes().split(b'\0')
            if len(argv)<2:continue
            path=Path(argv[1].decode())
            if not path.is_absolute():path=(p.parent/'cwd').resolve()/path
            if path.resolve()==target:matches.append(int(p.parent.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeDecodeError):pass
    return matches


def launch(script):
    with (OUT/(script.replace('.py','.log'))).open('a') as log:
        proc=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    return proc.pid


def inventory():
    primary=[]
    primary+=list((ROOT/'outputs/paper_completion_20261126/baseline').glob('*/H*/seed*/complete.json'))
    primary+=list((ROOT/'outputs/etth1_detail_search/confirm/L336/linear_skip').glob('H*/seed*/test_complete.json'))
    primary+=list((ROOT/'outputs/paper_seven_datasets_20261126').glob('*/*/H*/seed*/complete.json'))
    diagnostics=list(REVIEW.glob('*/*/H*/seed*/complete.json'))
    return primary,diagnostics


def check_once(force=False):
    primary,review=inventory();stage=read(ROOT/'outputs/paper_seven_datasets_20261126/progress.json')
    state=read(OUT/'state.json');last=read(OUT/'last_publication.json')
    signature=dict(primary=len(primary),review=len(review),review_complete=read(REVIEW/'progress.json').get('status')=='complete')
    result=dict(time=now(),primary_completed=len(primary),primary_total=420,review_completed_tasks=len(review),
        primary_status=stage.get('status','unknown'),publication='no_new_results')
    if force or signature!=state.get('last_published_signature') or last.get('status')!='published':
        with (OUT/'publication.log').open('a') as log:
            proc=subprocess.run([sys.executable,str(ROOT/'scripts/paper_publish_update.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode==0:
            state['last_published_signature']=signature;result['publication']='published'
            result['published_runs']=read(OUT/'last_publication.json').get('primary_published_runs')
        else:result.update(publication='failed_previous_pdf_preserved',failure=read(OUT/'last_failure.json'))
    state.update(last_check=now(),last_check_epoch=time.time(),last_result=result)
    atomic_json(OUT/'state.json',state)
    report=dict(last_check=state['last_check'],primary_completed=len(primary),primary_total=420,
        reviewer_progress=read(REVIEW/'progress.json'),last_publication=read(OUT/'last_publication.json'),
        next_hourly_check_epoch=state['last_check_epoch']+3600,model_api_calls=0)
    atomic_json(ROOT/'docs/validation/experiment_monitor_status.json',report)
    with (OUT/'git_backup.log').open('a') as log:
        proc=subprocess.run([sys.executable,str(ROOT/'scripts/paper_git_backup.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    result['git_backup']='pushed_verified' if proc.returncode==0 else 'failed_local_changes_preserved'
    state['last_result']=result;atomic_json(OUT/'state.json',state)
    with (OUT/'checks.jsonl').open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
    return result


def main():
    os.chdir(ROOT);OUT.mkdir(exist_ok=True)
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    if args.once:print(json.dumps(check_once(force=True),indent=2));return
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    lock=(OUT/'service.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    attempts=0;last_publication_attempt=0;transition_updated=False
    while True:
        primary=read(ROOT/'outputs/paper_seven_datasets_20261126/progress.json')
        review=read(REVIEW/'progress.json')
        if primary.get('status')=='complete' and review.get('status')!='complete' and not active('reviewer_experiments.py'):
            if attempts<3:
                attempts+=1;launch('reviewer_experiments.py');time.sleep(2)
            else:
                atomic_json(OUT/'reviewer_attention.json',dict(status='needs_attention',attempts=attempts,progress=review,time=now()))
        state=read(OUT/'state.json');due=time.time()-state.get('last_check_epoch',0)>=3600
        completed_transition=(primary.get('status')=='complete' and not transition_updated)
        all_complete_transition=review.get('status')=='complete' and not state.get('review_completion_published',False)
        if due or ((completed_transition or all_complete_transition) and time.time()-last_publication_attempt>=60):
            last_publication_attempt=time.time()
            try:result=check_once(force=completed_transition or all_complete_transition)
            except BaseException as error:
                result=dict(publication='failed_previous_pdf_preserved',error=str(error),time=now())
                atomic_json(OUT/'service_error.json',result)
            if result.get('publication') in ('published','no_new_results'):
                if primary.get('status')=='complete':transition_updated=True
                if review.get('status')=='complete':
                    state=read(OUT/'state.json');state['review_completion_published']=True;atomic_json(OUT/'state.json',state)
        status=dict(status='monitoring_primary' if primary.get('status')!='complete' else ('monitoring_review' if review.get('status')!='complete' else 'monitoring_complete_results'),
            time=now(),pid=os.getpid(),interval_seconds=3600,primary=primary,review=review,
            reviewer_controllers_and_workers=active('reviewer_experiments.py'),reviewer_launch_attempts=attempts,
            last_publication=read(OUT/'last_publication.json'),model_api_calls=0,
            git_backup=read(OUT/'git_backup.json'),
            next_hourly_check_epoch=read(OUT/'state.json').get('last_check_epoch',time.time())+3600)
        atomic_json(OUT/'status.json',status)
        # Keep the existing supervisor/dashboard contract alive until all work ends.
        atomic_json(LEGACY,dict(status=status['status'],hourly_monitor=str(OUT/'status.json'),primary=primary,review=review))
        time.sleep(30)


if __name__=='__main__':main()
