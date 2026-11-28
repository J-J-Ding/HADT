"""Read-only live progress dashboard for the frozen seven-dataset paper study."""
import html,json,time,re
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paper_revision_20261126'
BASE=ROOT/'outputs/paper_completion_20261126'
EXT=ROOT/'outputs/paper_seven_datasets_20261126'
REVIEW=ROOT/'outputs/reviewer_revision_20261128'
DATASETS=('ETTh1','ETTh2','ETTm1','ETTm2','Traffic','Weather','Electricity')
MODELS=('hadt','dlinear','patchtst','timemixer','itransformer')
def read(p):
    return json.loads(p.read_text()) if p.exists() else {}
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    while True:
        rows=[];done=0
        for d in DATASETS:
            for h in (96,192,336,720):
                cells=[f'<td>{d}</td><td>{h}</td>']
                for m in MODELS:
                    n=0
                    for s in (2024,2025,2026):
                        if d=='ETTh1' and m=='hadt':
                            p=ROOT/f'outputs/etth1_detail_search/confirm/L336/linear_skip/H{h}/seed{s}/test_complete.json'
                        elif d=='ETTh1':p=BASE/f'baseline/{m}/H{h}/seed{s}/complete.json'
                        else:p=EXT/f'{d}/{m}/H{h}/seed{s}/complete.json'
                        n+=p.exists()
                    done+=n
                    cells.append(f'<td class="{"complete" if n==3 else "pending"}">{n}/3</td>')
                rows.append('<tr>'+''.join(cells)+'</tr>')
        workers=[]
        for root in (BASE,EXT,REVIEW):
            for p in sorted((root/'workers').glob('gpu*.json')):
                if root==BASE and (EXT/'workers'/p.name).exists():continue
                if root==EXT and (REVIEW/'workers'/p.name).exists():continue
                r=read(p)
                if r.get('status')!='running':
                    state=r.get('status','unknown')
                    labels={'waiting':'等待领取任务','waiting_for_ett':'等待 ETT 全部完成','failed':'任务异常'}
                    label=labels.get(state,state)
                    previous=r.get('last_job',r.get('job',{}))
                    description=''
                    if previous:
                        description=f"；上一项：{previous.get('dataset','ETTh1')} / {previous.get('name',previous.get('variant',''))} / H{previous.get('horizon')} / seed{previous.get('seed')}"
                    workers.append(f'<li>{html.escape(p.stem)}: {html.escape(label+description)}</li>')
                    continue
                j=r.get('job',{});d=j.get('dataset','ETTh1');h=j.get('horizon');seed=j.get('seed')
                if root==BASE:
                    phase=j.get('phase')
                    name=('hadt_'+j['name']) if j.get('kind')=='ablation' else j.get('name','')
                    if phase=='screen':name+=f"_lr{j['lr']}"
                    run=root/f'{phase}/{name}/H{h}/seed{seed}'
                elif root==REVIEW:
                    directory=j['id'] if j['phase']=='validation_pilot' else j['variant']
                    run=root/j['phase']/directory/f'H{h}'/f'seed{seed}'
                else:run=root/f"{d}/{j['name']}/H{h}/seed{seed}"
                hist=run/'training_history.csv'
                epoch=max(0,len(hist.read_text().splitlines())-1) if hist.exists() else 0
                log=root/'workers'/(f"{j.get('phase')}_{p.stem}.log" if root==BASE else f'{p.stem}.log')
                latest=''
                if log.exists():
                    with log.open('rb') as stream:
                        stream.seek(max(0,log.stat().st_size-8192));tail=stream.read().decode(errors='replace')
                    bars=list(re.finditer(r'(Train [0-9]+/[0-9]+|Validation|Test [a-z_]+):[ ]*([0-9]+)%',tail))
                    if bars:latest='; latest: '+bars[-1].group(1)+' '+bars[-1].group(2)+'%'
                detail='复用 checkpoint 补充评估' if j.get('phase')=='diagnostic_reuse' else f'{epoch}/{j.get("epochs",30)} epochs completed'
                workers.append(f'<li>{html.escape(p.stem)}: {d}, {j.get("name",j.get("variant"))}, H{h}, seed{seed}; {detail}{html.escape(latest)}</li>')
        final=read(OUT/'finalization_progress.json')
        page='''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<style>body{font:15px system-ui;max-width:1080px;margin:32px auto;color:#234}table{border-collapse:collapse;width:100%}th,td{padding:7px 12px;border:1px solid #d9e2e8;text-align:center}.complete{background:#e1f4e7}.pending{background:#f6f7f9}progress{width:100%;height:22px}h1{font-size:24px}</style>'''
        page+=f'<h1>HADT-NQ 七数据集实验进度</h1><p>正式对比已完成 {done}/420 次测试。每个单元格需齐全三个种子。</p><progress value="{done}" max="420"></progress>'
        updated=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        page+=f'<p>更新时间：{updated}（北京时间）。</p>'
        monitor=read(ROOT/'outputs/paper_hourly_monitor/status.json')
        publication=monitor.get('last_publication',{})
        if monitor:
            next_check=datetime.fromtimestamp(monitor['next_hourly_check_epoch'],ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M')
            page+=f'<p>每小时自动更新论文；下次检查：{next_check}。最近更新：{html.escape(publication.get("updated","尚无"))}；论文纳入 {publication.get("primary_published_runs",0)} 次完整分组测试，{publication.get("pages","?")} 页。</p>'
        review=read(REVIEW/'progress.json')
        if review:
            page+=f'<p>V4 补实验：{html.escape(review.get("status","等待主对比完成"))}；正式训练 {review.get("formal_completed",0)}/36，预实验 {review.get("pilots_completed",0)}/3，复用模型补充评估 {review.get("reused_diagnostics_completed",0)}/12。</p>'
        priority=read(EXT/'scheduling_priority.json')
        if priority.get('status')=='draining_current_jobs':
            remaining=len(priority.get('current_remaining',[]))
            page+=f'<p>调度切换中：等待当前 {remaining} 个任务训练和测试完成，再启动 ETT 优先队列。已完成任务的 GPU 暂时等待。</p>'
        elif priority.get('status')=='ett_first_installed':
            page+='<p>ETT 优先：ETTh1、ETTh2、ETTm1、ETTm2 全部完成后，再启动新的 Traffic、Weather、Electricity 任务。</p>'
        nabl=len(list((BASE/'ablation').rglob('complete.json')))
        page+=f'<p>ETTh1消融：{nabl}/36。最终交付阶段：{html.escape(final.get("status","waiting"))}。</p><ul>'+''.join(workers)+'</ul>'
        page+='<table><thead><tr><th>Dataset</th><th>H</th>'+''.join(f'<th>{m}</th>' for m in MODELS)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table>'
        page+='<p>此页仅显示执行进度，不作为论文结果。正式表格只在完整实验矩阵核验通过后生成；每15秒自动刷新。</p>'
        (OUT/'progress.html').write_text(page)
        if final.get('status') in ('complete','failed'):return
        time.sleep(15)
if __name__=='__main__':main()
