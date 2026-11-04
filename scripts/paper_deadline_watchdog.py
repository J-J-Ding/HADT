"""Stop only the identified study process before the manuscript delivery reserve."""
import os,signal,time,json
from datetime import datetime
from pathlib import Path
pid=390953
proc=Path(f'/proc/{pid}')
original=(proc/'stat').read_text().split()[21]
cutoff=datetime.fromisoformat('2026-10-30T15:55:00+00:00').timestamp()
log=Path('outputs/paper_study/deadline_watchdog.json')
def same():
    try:return (proc/'stat').read_text().split()[21]==original and 'scripts/run_paper_experiments.py' in (proc/'cmdline').read_text()
    except FileNotFoundError:return False
while same() and time.time()<cutoff:time.sleep(min(5,max(.1,cutoff-time.time())))
if same():
    os.kill(pid,signal.SIGINT)
    log.write_text(json.dumps(dict(pid=pid,event='study cutoff SIGINT',unix=time.time())))
    time.sleep(10)
    if same():os.kill(pid,signal.SIGTERM)
else:log.write_text(json.dumps(dict(pid=pid,event='study completed before cutoff',unix=time.time())))
