"""Detached local supervisor: no network/model API calls; resumes frozen queues."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paper_revision_20261126'
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('paper_provenance',ROOT/'utils/provenance.py')
module=module_from_spec(spec);spec.loader.exec_module(module)
atomic_json=module.atomic_json
EXT=ROOT/'outputs/paper_seven_datasets_20261126'
def read(path):
    try:return json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:return {}
def active(script):
    matches=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if p.stat().st_uid!=os.getuid():continue
            argv=p.read_bytes().split(b'\0')
            if len(argv)<2:continue
            path=Path(argv[1].decode())
            cwd=(p.parent/'cwd').resolve()
            path=(cwd/path).resolve() if not path.is_absolute() else path.resolve()
            if path==(ROOT/'scripts'/script).resolve():matches.append(int(p.parent.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError,UnicodeDecodeError):pass
    return matches
def launch(script):
    log=(OUT/('background_'+script.replace('.py','.log'))).open('a')
    proc=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script)],cwd=ROOT,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True)
    log.close()
    return proc.pid
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    lock=(OUT/'supervisor.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    attempts={'extend_paper_datasets.py':0,'finalize_completed_paper.py':0}
    while True:
        training=read(EXT/'progress.json');final=read(OUT/'finalization_progress.json')
        failed_workers=[str(p) for p in (EXT/'workers').glob('gpu*.json') if read(p).get('status')=='failed']
        stage='running'
        if training.get('status')!='complete' and not active('extend_paper_datasets.py'):
            script='extend_paper_datasets.py'
            if attempts[script]>=3:
                atomic_json(OUT/'supervisor_status.json',dict(status='needs_attention',reason='Training controller stopped after three automatic recovery attempts.',failed_workers=failed_workers));return
            attempts[script]+=1;launch(script)
            # Existing completed runs are hash-verified and skipped by the controller.
            time.sleep(10)
        if final.get('status')=='complete':
            atomic_json(OUT/'supervisor_status.json',dict(status='complete',model_api_calls=0,pdf='HADT_NQ_ICCRD2027_LaTeX/main.pdf',recovery_attempts=attempts));return
        if not active('finalize_completed_paper.py') and (not failed_workers or training.get('status')=='complete'):
            script='finalize_completed_paper.py'
            if attempts[script]>=3:
                atomic_json(OUT/'supervisor_status.json',dict(status='needs_attention',reason='Automatic paper finalization failed; inspect finalize_step logs.',recovery_attempts=attempts));return
            attempts[script]+=1;launch(script)
        if not active('watch_paper_progress.py') and final.get('status')!='failed':
            launch('watch_paper_progress.py')
        atomic_json(OUT/'supervisor_status.json',dict(status=stage,pid=os.getpid(),session_id=os.getsid(0),
            training_controllers_and_workers=active('extend_paper_datasets.py'),
            finalizers=active('finalize_completed_paper.py'),dashboards=active('watch_paper_progress.py'),
            recovery_attempts=attempts,model_api_calls=0))
        time.sleep(30)
if __name__=='__main__':main()
