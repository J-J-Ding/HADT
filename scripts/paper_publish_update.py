"""Stage, validate and atomically publish new complete result groups each hour."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.provenance import atomic_json,sha256
PAPER=ROOT/'HADT_NQ_ICCRD2027_LaTeX'
OUT=ROOT/'outputs/paper_hourly_monitor'


def read(p):return json.loads(Path(p).read_text())


def intro(text):return text.split(r'\section{Introduction}',1)[1].split(r'\input{sections/related_work}',1)[0]


def result_scope(stage,audit):
    datasets=audit['datasets'];settings=audit['complete_settings'];n=len(settings);runs=audit['published_runs'];best=audit['best']
    names={1:'one',2:'two',3:'three',4:'four',5:'five',6:'six',7:'seven'}
    description=names[len(datasets)]+(' ETT' if all(d.startswith('ETT') for d in datasets) else ' benchmark')+' datasets'
    (stage/'results/completion_scope.tex').write_text(
        f'This revision reports {runs} completed test runs across {description} and {n} dataset--horizon settings. '
        'Each displayed setting contains five models and three seeds. '
        +('The seven-dataset matrix is complete. ' if runs==420 else 'Other settings in the seven-dataset protocol remain in progress and are excluded until complete. ')
        +'Table~\\ref{tab:data} describes the datasets represented by these comparisons.\n')
    metadata={'ETTh1':(17420,7,'60/20/20'),'ETTh2':(17420,7,'60/20/20'),
        'ETTm1':(69680,7,'60/20/20'),'ETTm2':(69680,7,'60/20/20'),
        'Traffic':(17544,862,'70/10/20'),'Weather':(52696,21,'70/10/20'),'Electricity':(26304,321,'70/10/20')}
    from scripts import style_paper_tables as styles
    styles.write_table('completion_dataset_table.tex','Datasets represented in the completed comparisons.','tab:data',
        ['Dataset','Rows','Variables','Split'],
        [' & '.join([d,f'{metadata[d][0]:,}',str(metadata[d][1]),metadata[d][2]])+r' \\' for d in datasets],'lrrl')
    remaining=[d for d in metadata if sum(x==d for x,h in settings)<4]
    scope=f'The current comparison includes {description}. '
    if remaining:scope+='The remaining settings on '+', '.join(remaining)+' await complete model and seed groups. '
    scope+='Four ETT series are related benchmarks, so their results alone do not establish generalization to all domains.\n'
    (stage/'results/completion_limitations.tex').write_text(scope)
    # Change only quantitative result statements outside Introduction.
    path=stage/'main.tex';text=path.read_text();original_intro=intro(text)
    a=text.index(r'\section{Introduction}');b=text.index(r'\input{sections/related_work}',a)
    before=text[:a];after=text[b:]
    before=re.sub(r'Across \d+ completed dataset--horizon settings on [^,]+ datasets, HDA-NQ attains the lowest mean pinball loss in \d+ settings',
        f'Across {n} completed dataset--horizon settings on {description}, HDA-NQ attains the lowest mean pinball loss in {best} settings',before)
    after=re.sub(r'The (?:completed|complete) comparisons on [^.]+? datasets favor the proposed model in \d+ of \d+ dataset--horizon settings',
        f'The completed comparisons on {description} favor the proposed model in {best} of {n} dataset--horizon settings',after)
    if runs==420:after=after.replace(' Conclusions about Traffic, Weather, and Electricity await their complete experimental matrices.','')
    updated=before+text[a:b]+after;assert intro(updated)==original_intro;path.write_text(updated)
    p=stage/'sections/experiments.tex';s=p.read_text()
    s=s.replace('The completed comparisons reported here cover four related ETT datasets. The broader benchmark protocol also includes Traffic, Weather, and Electricity; their incomplete comparisons are excluded from this revision.',r'\input{results/completion_limitations}')
    p.write_text(s)


def validate(stage):
    dependencies=set()
    def expand(path):
        path=path.resolve()
        assert path not in dependencies,('Repeated TeX dependency',path)
        dependencies.add(path);text=path.read_text()
        for name in re.findall(r'\\input\{([^}]+)\}',text):
            target=stage/(name if name.endswith('.tex') else name+'.tex')
            assert target.is_file(),target
            text=text.replace(r'\input{'+name+'}',expand(target))
        return text
    main=expand(stage/'main.tex');dependencies.clear();supp=expand(stage/'supplement.tex')
    keys=re.findall(r'@\w+\s*\{\s*([^,\s]+)',(stage/'references.bib').read_text())
    duplicate_keys=sorted({k for k in keys if keys.count(k)>1})
    for name,text in [('main',main),('supplement',supp)]:
        labels=re.findall(r'\\label\{([^}]+)\}',text);assert len(labels)==len(set(labels))
        assert set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',text))<=set(labels)
        cites={k for group in re.findall(r'\\cite\{([^}]+)\}',text) for k in group.split(',')};assert cites<=set(keys)
        for f in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):assert (stage/f).is_file()
        log=(stage/f'{name}.log').read_text(errors='replace')
        assert not re.search(r'LaTeX Error|Emergency stop|Missing character|Float too large|Citation .*undefined|Reference .*undefined|There were undefined',log,re.I),name
        overflow=[float(x) for x in re.findall(r'Overfull \\hbox \(([0-9.]+)pt',log)]
        assert max(overflow,default=0)<=3,(name,overflow)
        plain=subprocess.check_output(['pdftotext',str(stage/f'{name}.pdf'),'-'],text=True);assert '??' not in plain
    info=subprocess.check_output(['pdfinfo',str(stage/'main.pdf')],text=True)
    pages=int(re.search(r'Pages:\s+(\d+)',info).group(1));assert pages<=10,('Page limit',pages)
    fonts=subprocess.check_output(['pdffonts',str(stage/'main.pdf')],text=True);assert 'Type 3' not in fonts
    return dict(pages=pages,undefined_citations=0,undefined_references=0,template_changes=False,
        bibliography_duplicate_keys=duplicate_keys,bibliography_policy='Preserve author edits; report duplicate keys without rewriting bibliography. Compilation and citation resolution must succeed.')


def publish():
    os.chdir(ROOT);OUT.mkdir(exist_ok=True)
    lock=(OUT/'publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    work=OUT/'updates'/stamp;stage=work/'paper';stage.parent.mkdir(parents=True,exist_ok=True)
    inputs=[p for p in PAPER.rglob('*') if p.is_file() and p.suffix in ('.tex','.bib','.cls','.bst')]
    inputs+=list((PAPER/'figures').glob('*'))
    fingerprints={str(p.relative_to(PAPER)):sha256(p) for p in inputs if p.is_file()}
    original_main=(PAPER/'main.tex').read_text();original_intro=intro(original_main)
    shutil.copytree(PAPER,stage,ignore=shutil.ignore_patterns('*.zip','*.aux','*.log','*.blg','*.bbl','*.out','main.pdf','supplement.pdf'))
    from scripts import publish_completed_paper_snapshot as snapshot
    from scripts import style_paper_tables as styles
    snapshot.P=stage/'results';styles.P=stage/'results'
    snapshot.main()
    audit=read(stage/'results/snapshot_audit.json');result_scope(stage,audit)
    review_complete=read(ROOT/'outputs/reviewer_revision_20261128/progress.json').get('status')=='complete' if (ROOT/'outputs/reviewer_revision_20261128/progress.json').exists() else False
    if review_complete:
        from scripts.reviewer_publish_results import publish as publish_review
        assert publish_review(stage,ROOT/'outputs/reviewer_revision_20261128')
        p=stage/'sections/experiments.tex';s=p.read_text()
        if r'\input{results/reviewer_summary}' not in s:s=s.replace(r'\subsection{Limitations}',r'\input{results/reviewer_summary}'+'\n\n'+r'\subsection{Limitations}')
        p.write_text(s)
        p=stage/'supplement.tex';s=p.read_text()
        if r'\input{results/reviewer_supplement}' not in s:s=s.replace(r'\end{document}',r'\input{results/reviewer_supplement}'+'\n'+r'\end{document}')
        p.write_text(s)
    for name in ('main','supplement'):
        env=os.environ.copy();env['TECTONIC_OFFLINE']='1'
        with (work/f'build_{name}.log').open('w') as log:
            subprocess.run(['bash',str(ROOT/'scripts/build_paper.sh'),str(stage/f'{name}.tex')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt=validate(stage)
    assert intro((stage/'main.tex').read_text())==original_intro
    # Refuse to overwrite any edit made by the user during staging/compilation.
    for relative,digest in fingerprints.items():assert sha256(PAPER/relative)==digest,('User input changed during update',relative)
    candidates=[stage/'main.tex',stage/'sections/experiments.tex',stage/'supplement.tex']
    candidates+=list((stage/'results').glob('*'))
    if review_complete:candidates+=list((stage/'figures').glob('reviewer_*'))
    for name in ('main','supplement'):
        candidates+=[stage/f'{name}{suffix}' for suffix in ('.pdf','.log','.aux','.bbl','.blg','.out') if (stage/f'{name}{suffix}').exists()]
    changed=[]
    for source in candidates:
        if not source.is_file():continue
        rel=source.relative_to(stage);target=PAPER/rel
        if target.exists() and sha256(target)==sha256(source):continue
        if target.exists():
            backup=work/'before'/rel;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,backup)
        temp=target.with_name(target.name+'.hourly.tmp');shutil.copy2(source,temp);temp.replace(target);changed.append(str(rel))
    receipt.update(status='published',updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        primary_published_runs=audit['published_runs'],complete_settings=len(audit['complete_settings']),best_count=audit['best'],
        reviewer_results_included=review_complete,introduction_unchanged=True,changed_files=changed,backup=str(work/'before'),
        pdf_sha256=sha256(PAPER/'main.pdf'))
    atomic_json(work/'receipt.json',receipt);atomic_json(OUT/'last_publication.json',receipt)
    shutil.rmtree(stage)
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=='__main__':
    try:publish()
    except BaseException as error:
        OUT.mkdir(exist_ok=True);atomic_json(OUT/'last_failure.json',dict(status='publication_failed_previous_pdf_preserved',error=str(error) or repr(error),
            time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()));raise
