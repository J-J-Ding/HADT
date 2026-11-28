"""Validate compiled manuscript dependencies and package a self-contained LaTeX delivery."""
import hashlib,json,re,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'HADT_NQ_ICCRD2027_LaTeX'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
dependencies=set()
def expand(path):
    path=path.resolve()
    if path in dependencies:return ''
    dependencies.add(path);text=path.read_text()
    text=re.sub(r'(?<!\\)%[^\n]*','',text)
    for include in re.findall(r'\\input\{([^}]+)\}',text):
        target=P/(include if include.endswith('.tex') else include+'.tex')
        assert target.is_file(),f'Missing TeX input: {target}'
        text=text.replace(r'\input{'+include+'}',expand(target))
    return text
text=expand(P/'main.tex')
supplement_text=expand(P/'supplement.tex')
assert not re.search(r'\bshort[- ](?:term|horizon|range)\b',text+supplement_text,re.I), 'Forecasting scope must remain long-term'
bib=(P/'references.bib').read_text()
keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bib))
cites=set()
for group in re.findall(r'\\cite(?:\[[^\]]*\])?\{([^}]+)\}',text):cites.update(group.split(','))
assert cites<=keys,cites-keys
assert {'krawczyk2025financial','vidal2026hybrid','abisoye2024solar'}<=cites
assert len(keys)==len(re.findall(r'@\w+\s*\{',bib)),'Duplicate bibliography keys'
labels=re.findall(r'\\label\{([^}]+)\}',text)
assert len(labels)==len(set(labels)),'Duplicate labels'
references=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',text)
assert set(references)<=set(labels),set(references)-set(labels)
assert not re.search(r'\b(TODO|TBD|placeholder|pending|unconfirmed)\b',text,re.I)
figures=[]
for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
    f=P/name;assert f.is_file() and f.stat().st_size>1000
    info=subprocess.check_output(['pdfinfo',str(f)],text=True)
    assert re.search(r'Pages:\s+1\b',info),f
    figures.append(f)
log=(P/'main.log').read_text(errors='replace')
for pattern in [r'LaTeX Error',r'Emergency stop',r'Missing character',r'Citation .*undefined',r'Reference .*undefined',r'There were undefined']:
    assert not re.search(pattern,log,re.I),pattern
supp_log=(P/'supplement.log').read_text(errors='replace')
assert not re.search(r'LaTeX Error|Missing character|There were undefined|Emergency stop',supp_log,re.I)
pdf=P/'main.pdf'
info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
assert 'Anonymous Author(s)' in info
pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
assert pages<=10, f'Main manuscript including references exceeds 10 pages: {pages}'
plain=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
assert '??' not in plain
assert all(t in plain for t in ['Introduction','Related Work','Method','Experiments','Conclusion'])
assert all(t in plain for t in ['Krawczyk','Vidal','Abisoye'])
fonts=subprocess.check_output(['pdffonts',str(pdf)],text=True)
assert 'Type 3' not in fonts
for row in fonts.splitlines()[2:]:
    fields=row.split()
    if fields:assert fields[-5]=='yes',f'Unembedded font: {row}'
completion=json.loads((P/'results/completion_audit.json').read_text())
assert completion['status']=='passed' and completion['new_baseline_runs']==336 and completion['new_ablation_runs']==36
assert completion['datasets']==['ETTh1','ETTh2','ETTm1','ETTm2','Traffic','Weather','Electricity'] and completion['primary_test_runs']==420
overfull=[float(x) for x in re.findall(r'Overfull \\hbox \(([0-9.]+)pt',log)]
assert max(overfull,default=0)<=3, 'Text or table overflow requires layout review'
receipt=dict(status='passed',date='2026-11-26',pages=pages,cited_references=len(cites),
             main_page_limit_including_references=10,
             all_three_requested_references_cited=True,undefined_citations=0,undefined_references=0,
             missing_characters=0,placeholder_tokens=0,figure_count=len(figures),
             figures_are_single_complete_pages=True,type3_fonts=False,all_fonts_embedded=True,
             maximum_overfull_hbox_pt=max(overfull,default=0),new_training_jobs=456,
             validation=json.loads((P/'results/completion_audit.json').read_text()),
             manuscript_source_sha256={str(p.relative_to(P)):sha(p) for p in sorted(dependencies)},
             figure_sha256={str(f.relative_to(P)):sha(f) for f in figures},
             pdf_sha256=sha(pdf),bibliography_sha256=sha(P/'references.bib'))
(P/'results/revision_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
# The manuscript package includes exactly the active source and its artwork, not old draft packages.
extra=[P/'references.bib',P/'acmart.cls',P/'ACM-Reference-Format.bst',P/'README_CN.md',
       P/'SUBMISSION_CHECKLIST.md',P/'supplement.pdf',pdf]
extra.extend((P/'results').glob('completion_*.csv'))
extra.extend((P/'results').glob('completion_*protocol.json'))
extra.extend([P/'results/completion_selection.json',P/'results/completion_model_checks.json'])
extra.extend([P/'results/completion_audit.json',P/'results/revision_sources.json',
              P/'results/revision_validation.json',P/'results/reference_additions_20261126.json',
              P/'results/forecast_example_selection.json',P/'results/forecast_example_selection.csv',
              P/'results/revised_attention_data.npz'])
files=set(dependencies)|set(figures)|set(extra)
for f in figures:
    for ext in ('.svg','.png'):
        if f.with_suffix(ext).exists():files.add(f.with_suffix(ext))
archive=P/'HADT_NQ_complete_revision_20261126.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for f in sorted(files):
        assert f.is_file(),f
        z.write(f,str(f.relative_to(P)))
print(json.dumps(dict(status='passed',pages=pages,references=len(cites),figures=len(figures),package=str(archive)),indent=2))
