"""Render saved, fixed-window predictions on CPU and build a local image gallery."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import html
import json
import math
import multiprocessing
import os
from pathlib import Path
from urllib.parse import quote

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 9})
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'outputs/architecture_study'
DEFAULT_GALLERY = Path(__file__).resolve().parents[1] / '预测结果浏览.html'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_text(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(value, encoding='utf-8')
    temp.replace(path)


def validate_arrays(data):
    history, target, q = data['history'], data['target'], data['quantiles']
    if history.ndim != 3 or target.ndim != 3 or q.shape != (*target.shape, 7):
        raise ValueError('Expected history [S,L,C], target [S,H,C], quantiles [S,H,C,7]')
    s, h, c = target.shape
    if min(s, h, c) < 1 or history.shape[0] != s or history.shape[2] != c:
        raise ValueError('History and prediction dimensions disagree')
    if len(data['sample_indices']) != s or len(data['channel_indices']) != c or len(data['feature_columns']) != c:
        raise ValueError('Sample/channel labels disagree with arrays')
    if not all(np.isfinite(a).all() for a in [history, target, q]) or (np.diff(q, axis=-1) < -1e-10).any():
        raise ValueError('Non-finite data or crossed quantiles; refusing to draw')


def panel(ax, data, window, channel, history_steps=96):
    history = data['history'][window, -history_steps:, channel]
    truth = data['target'][window, :, channel]
    q = data['quantiles'][window, :, channel]
    future = np.arange(len(truth))
    ax.plot(np.arange(-len(history), 0), history, color='#777777', lw=1, label='Observed history')
    ax.fill_between(future, q[:, 0], q[:, 6], color='#376b87', alpha=.15, label='90% prediction interval')
    ax.fill_between(future, q[:, 2], q[:, 4], color='#376b87', alpha=.27, label='50% prediction interval')
    ax.plot(future, truth, color='#202020', lw=1.15, label='Observed future')
    ax.plot(future, q[:, 3], color='#17679a', lw=1.15, label='Forecast median')
    ax.axvline(0, ls='--', color='#777777', lw=.8)
    ax.grid(alpha=.15)
    ax.set_xlabel('Step relative to forecast start')
    ax.set_ylabel('Original units')


def render_run(record):
    run = Path(record['source_dir'])
    output = run / 'figures/predictions'
    output.mkdir(parents=True, exist_ok=True)
    source = run / 'forecast_samples.npz'
    if not source.is_file():
        source = run / 'figures/forecast_samples.npz'
    signature = dict(source_sha256=digest(source), renderer_sha256=digest(__file__))
    marker = output / 'plot_complete.json'
    with (output / '.plot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            saved = json.loads(marker.read_text())
            if all(saved.get(k) == v for k, v in signature.items()) and all((output / p).is_file() for p in saved['files']):
                return dict(record, plot_dir=str(output))
        with np.load(source, allow_pickle=False) as raw:
            data = {k: raw[k] for k in raw.files}
        validate_arrays(data)
        names = [str(n) for n in data['feature_columns']]
        channels = [int(n) for n in data['channel_indices']]
        windows = [int(n) for n in data['sample_indices']]
        title = f"{record['dataset']} | H={record['horizon']} | {record['candidate_id']} | seed={record['seed']}"
        files = ['overview.png', 'index.html']
        # The first eligible test window is fixed before looking at forecast quality.
        fig, axes = plt.subplots(math.ceil(len(names) / 2), 2, figsize=(14, 3.0 * math.ceil(len(names) / 2)), squeeze=False)
        try:
            for j, ax in enumerate(axes.flat):
                if j >= len(names):
                    ax.axis('off')
                    continue
                panel(ax, data, 0, j)
                ax.set_title(f'{names[j]} (original channel {channels[j]})')
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8)
            fig.suptitle(title + f'\nFirst fixed test window: {windows[0]}; each variable has its own original-unit scale', fontsize=11)
            fig.tight_layout(rect=(0, .055, 1, .94))
            fig.savefig(output / 'overview.png', dpi=135)
        finally:
            plt.close(fig)
        entries = []
        for j, channel in enumerate(channels):
            basename = f'channel_{channel:03d}'
            fig, axes = plt.subplots(len(windows), 1, figsize=(13, 2.7 * len(windows)), squeeze=False)
            try:
                for i, window in enumerate(windows):
                    panel(axes[i, 0], data, i, j)
                    axes[i, 0].set_title(f'Fixed test window {window}')
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8)
                fig.suptitle(title + f'\nVariable: {names[j]} (original channel {channel})', fontsize=11)
                fig.tight_layout(rect=(0, .05, 1, .94))
                for extension in ['png', 'pdf']:
                    name = basename + '.' + extension
                    fig.savefig(output / name, dpi=140)
                    files.append(name)
            finally:
                plt.close(fig)
            entries.append(f'<section><h2>{html.escape(names[j])} · channel {channel}</h2>'
                           f'<a href="{basename}.pdf">下载 PDF</a> · <a href="{basename}.png">打开 PNG</a>'
                           f'<img loading="lazy" src="{basename}.png" alt="三个固定窗口的预测曲线"></section>')
        atomic_text(output / 'index.html', '<!doctype html><meta charset="utf-8"><title>' + html.escape(title) + '</title>'
                    '<style>body{max-width:1400px;margin:30px auto;font:16px sans-serif;padding:20px;background:#fafafa}'
                    'img{display:block;width:100%;height:auto}section{margin:28px 0;background:white;padding:16px}</style>'
                    '<h1>' + html.escape(title) + '</h1><p>真实值（黑）、预测中位数（蓝）、50%/90%预测区间。'
                    '显示最近96步历史；首、中、末测试窗口固定选取，各面板纵轴单位为原始量纲。'
                    '图示不替代全部测试窗口的指标。</p>' + ''.join(entries))
        atomic_text(marker, json.dumps(dict(**signature, sample_indices=windows, channel_indices=channels,
                    feature_columns=names, files=files, scope='All saved variables, fixed first/middle/last test windows.'), indent=2))
    return dict(record, plot_dir=str(output))


def write_gallery(records, gallery=DEFAULT_GALLERY):
    gallery = Path(gallery)
    gallery.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    available = []
    for r in records:
        output = Path(r['source_dir']) / 'figures/predictions'
        if not (output / 'plot_complete.json').exists():
            continue
        relative = quote(os.path.relpath(output, gallery.parent).replace(os.sep, '/'), safe='/')
        label = f"{r['dataset']} · H{r['horizon']} · {r['candidate_id']} · seed{r['seed']}"
        attributes = ' '.join(f'data-{key}="{html.escape(str(r[key]), quote=True)}"' for key in ['dataset', 'horizon', 'candidate_id', 'seed'])
        evaluation_link = ''
        if (Path(r['source_dir']) / 'evaluation.html').is_file():
            evaluation_url = quote(os.path.relpath(Path(r['source_dir']) / 'evaluation.html', gallery.parent).replace(os.sep, '/'), safe='/')
            evaluation_link = f'<p><a href="{evaluation_url}">查看完整测试指标和简单基线对比</a></p>'
        cards.append(f'<article {attributes}><h2>{html.escape(label)}</h2><a href="{relative}/index.html">'
                     f'<img loading="lazy" src="{relative}/overview.png" alt="{html.escape(label, quote=True)}">'
                     '</a><p><a href="' + relative + '/index.html">查看各变量的首、中、末窗口曲线和 PDF</a></p>'
                     + evaluation_link + '</article>')
        available.append(r)
    selectors = []
    for key, label in [('dataset', '数据集'), ('horizon', '预测长度'), ('candidate_id', '模型'), ('seed', '随机种子')]:
        values = sorted({str(r[key]) for r in available})
        options = ''.join('<option>' + html.escape(v) + '</option>' for v in values)
        selectors.append(f'<label>{label} <select data-key="{key}"><option value="">全部</option>{options}</select></label>')
    atomic_text(gallery, '<!doctype html><meta charset="utf-8"><title>实验预测曲线浏览</title>'
                '<style>body{font:16px sans-serif;background:#f5f7fa;max-width:1500px;margin:30px auto;padding:20px}'
                '.filters{display:flex;flex-wrap:wrap;gap:18px;position:sticky;top:0;background:#f5f7fa;padding:12px 0}'
                'select{padding:8px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}'
                'article{background:white;padding:15px;border:1px solid #dde2e8}h2{font-size:17px}img{width:100%;height:auto}'
                '@media(max-width:550px){main{display:block}article{margin-bottom:20px}}</style>'
                '<h1>实验预测曲线浏览</h1><p>已生成 ' + str(len(available)) + ' 次完整测试的预测图片。'
                '每次展示已保存的变量（数量随实验设置）、固定首/中/末测试窗口；不是全部滑动窗口。'
                '黑线为真实未来值，蓝线为中位数，阴影为50%/90%预测区间，灰线为历史值。点击图片查看各变量详情。</p>'
                '<div class="filters">' + ''.join(selectors) + '</div><p id="count"></p><main>' + ''.join(cards) + '</main>'
                '<script>const filters=[...document.querySelectorAll("select")],cards=[...document.querySelectorAll("article")];'
                'function apply(){let n=0;for(const c of cards){let ok=filters.every(f=>!f.value||c.getAttribute("data-"+f.dataset.key)===f.value);'
                'c.hidden=!ok;if(ok)n++;}document.getElementById("count").textContent="当前显示 "+n+" 次实验";}'
                'filters.forEach(f=>f.addEventListener("change",apply));apply();</script>')
    return len(available)


def render_completed(records, gallery=DEFAULT_GALLERY, workers=1):
    if workers == 1:
        for record in records:
            render_run(record)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = [executor.submit(render_run, record) for record in records]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                if i == 1 or i % 5 == 0:
                    write_gallery(records, gallery)
                    print(f'PLOTS {i}/{len(records)}', flush=True)
    return write_gallery(records, gallery)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--gallery', type=Path, default=DEFAULT_GALLERY)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--extra-root', type=Path, action='append', default=[],
                        help='Also include standard main.py runs under this output directory; repeatable.')
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in sorted(args.root.glob('*/*/*/H*/seed*/complete.json'))]
    for root in args.extra_root:
        for path in sorted(root.rglob('resolved_config.json')):
            run = path.parent
            if not (run / 'metrics.json').is_file() or not (run / 'figures/forecast_samples.npz').is_file():
                continue
            resolved = json.loads(path.read_text())
            records.append(dict(source_dir=str(run.resolve()), dataset=resolved['dataset'],
                                horizon=resolved['pred_len'], seed=resolved['seed'],
                                candidate_id=resolved['config']['project']['name']))
    count = render_completed(records, args.gallery, args.workers)
    print(f'Generated plots for {count} completed runs; gallery: {args.gallery}', flush=True)


if __name__ == '__main__':
    main()
