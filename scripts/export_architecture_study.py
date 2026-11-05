"""Incrementally synchronize measured architecture-study artifacts into the manuscript."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans'})

ROOT = Path('outputs/architecture_study')
PAPER = Path('HADT_NQ_ICCRD2027_LaTeX')
OUT = PAPER / 'results'
FIG = PAPER / 'figures'
SEEDS = {2024, 2025, 2026}
LABELS = dict(transformer3='Transformer-3 (original)', identity='No encoder', mlp3='MLP-3',
              cnn3='CNN-3', cnn_transformer='CNN + Transformer', transformer1='Transformer-1',
              cnn64_lr3e4='CNN-3, D64, lr 3e-4', mlp64_lr3e4='MLP-3, D64, lr 3e-4',
              linear='Linear-NQ', patch_direct='Patch-direct', mlp_direct='MLP-direct', cnn_direct='CNN-direct')


def atomic_text(path, text):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def records(kind):
    return [json.loads(p.read_text()) for p in sorted(ROOT.glob(f'*/*/*/H*/seed*/{kind}.json'))]


def groups(rows):
    result = []
    if not rows:
        return result
    frame = pd.DataFrame(rows)
    for keys, group in frame.groupby(['dataset', 'horizon', 'candidate_id']):
        if set(group.seed) != SEEDS or len(group) != 3:
            continue
        row = dict(dataset=keys[0], horizon=int(keys[1]), candidate_id=keys[2], n=3)
        for metric in ['best_val_pinball', 'QuantileLoss_standardized', 'PICP90', 'PINAW90', 'NormalizedWinkler90']:
            if metric in group:
                row[metric + '_mean'] = float(group[metric].mean())
                row[metric + '_std'] = float(group[metric].std(ddof=1))
        result.append(row)
    return result


def table(caption, label, header, rows, alignment):
    return '\n'.join([r'\begin{table*}[t]', r'\centering\small',
                      '\\caption{' + caption + '}', '\\label{' + label + '}',
                      '\\begin{tabular}{@{}' + alignment + '@{}}', r'\toprule',
                      ' & '.join(header) + r' \\', r'\midrule', *rows,
                      r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


def pm(row, metric, scale=1, digits=5):
    return f"${row[metric + '_mean'] * scale:.{digits}f}\\pm{row[metric + '_std'] * scale:.{digits}f}$"


def save_figure(fig, name):
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(FIG / (name + '.png'), bbox_inches='tight', dpi=180)
    plt.close(fig)


def plots(screen, confirmation, selected, evaluated):
    if screen:
        ordered = sorted(screen, key=lambda r: (r['best_val_pinball'], r['candidate_id']))
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        positions = np.arange(len(ordered))
        ax.barh(positions, [r['best_val_pinball'] for r in ordered], color='#376b87')
        ax.set_yticks(positions, [LABELS[r['candidate_id']] for r in ordered], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('ETTh1 validation pinball (8-epoch screening, seed 2024)')
        ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        save_figure(fig, 'architecture_screening')
    if confirmation:
        frame = pd.DataFrame(confirmation)
        fig, ax = plt.subplots(figsize=(7.2, 3))
        for name, group in frame.groupby('candidate_id'):
            ax.scatter([LABELS[name]] * len(group), group.best_val_pinball, s=20, color='#376b87')
        ax.set_ylabel('Best validation pinball')
        ax.tick_params(axis='x', labelsize=7)
        ax.set_title('30-epoch confirmation; dots are completed training seeds')
        ax.grid(axis='y', alpha=.2)
        fig.tight_layout()
        save_figure(fig, 'architecture_confirmation')
        names = sorted(frame.candidate_id.unique())
        fig, axes = plt.subplots(1, len(names), figsize=(7.2, 2.6), squeeze=False, sharey=True)
        colors = {2024: '#376b87', 2025: '#a16a17', 2026: '#75486d'}
        for ax, name in zip(axes[0], names):
            for r in confirmation:
                if r['candidate_id'] != name:
                    continue
                history = pd.read_csv(Path(r['source_dir']) / 'training_history.csv')
                ax.plot(history.epoch.to_numpy(), history.val_pinball.to_numpy(),
                        color=colors[r['seed']], label=str(r['seed']), lw=1)
            ax.set_title(LABELS[name], fontsize=8)
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.2)
        axes[0, 0].set_ylabel('Validation pinball')
        axes[0, -1].legend(fontsize=7)
        fig.tight_layout()
        save_figure(fig, 'architecture_learning_curves')
    if selected:
        candidate = selected['candidate']
        fig, ax = plt.subplots(figsize=(7.2, 1.9))
        ax.axis('off')
        encoder = dict(transformer='Temporal Transformer', cnn='Dilated temporal CNN',
                       mlp='Per-token residual MLP', hybrid='CNN + Transformer', identity='Bypass encoder')[candidate['encoder']]
        if candidate['decoder'] == 'linear':
            blocks = ['History\nL=336', 'RevIN', 'Linear\n336 to 7H', 'Median +\npositive gaps', 'Reverse RevIN\n7 quantiles']
        elif candidate['decoder'] == 'mlp_direct':
            blocks = ['History\nL=336', 'RevIN', 'MLP\n336 to 256 to 7H', 'Median +\npositive gaps', 'Reverse RevIN\n7 quantiles']
        else:
            decoder = 'Horizon deformable\nattention, K=8' if candidate['decoder'] == 'deformable' else 'Flatten + direct\n7H projection'
            depth_text = '2 CNN + 1 Transformer' if candidate['encoder'] == 'hybrid' else str(candidate['depth']) + ' blocks'
            blocks = ['RevIN + patches\nP16 / S8', encoder + '\n' + depth_text,
                      decoder, 'Median +\npositive gaps', 'Reverse RevIN\n7 quantiles']
        for i, text in enumerate(blocks):
            x = i * 2
            ax.text(x, .5, text, ha='center', va='center', fontsize=7.3,
                    bbox=dict(boxstyle='round,pad=.6', facecolor='#e9f2f5', edgecolor='#376b87'))
            if i < len(blocks) - 1:
                ax.annotate('', xy=(x + 1.25, .5), xytext=(x + .75, .5), arrowprops=dict(arrowstyle='->'))
        ax.set(xlim=(-1.1, 9.1), ylim=(-.2, 1.4))
        ax.set_title('Validation-selected candidate: ' + LABELS[selected['candidate_id']], fontsize=10)
        fig.tight_layout()
        save_figure(fig, 'architecture_selected')
        # First eligible test window, last variable; selection is frozen before test scores exist.
        examples = [r for r in evaluated if r['candidate_id'] == selected['candidate_id']
                    and r['dataset'] == 'ETTh1' and r['horizon'] == 96 and r['seed'] == 2024]
        if examples:
            data = np.load(Path(examples[0]['source_dir']) / 'forecast_samples.npz')
            fig, ax = plt.subplots(figsize=(7.2, 2.6))
            q = data['quantiles'][0, :, -1]
            ax.plot(data['target'][0, :, -1], color='#252525', label='Observed')
            ax.plot(q[:, 3], color='#376b87', label='Selected median')
            ax.fill_between(np.arange(96), q[:, 0], q[:, 6], color='#376b87', alpha=.2, label='90% interval')
            ax.set(xlabel='Forecast step', ylabel=str(data['feature_columns'][-1]))
            ax.legend(ncol=3, fontsize=8)
            fig.tight_layout()
            save_figure(fig, 'architecture_forecast')


def verify(rows):
    from scripts.run_architecture_study import sha
    checked = []
    hashes = {}
    protocol = json.loads((ROOT / 'protocol.json').read_text()) if (ROOT / 'protocol.json').exists() else None
    def cached_sha(path):
        key = str(path)
        if key not in hashes:
            hashes[key] = sha(path)
        return hashes[key]
    for row in rows:
        path = Path(row['source_dir'])
        history = pd.read_csv(path / 'training_history.csv')
        observed = history.loc[history.epoch == row['best_epoch'], 'val_pinball'].iloc[0]
        if not np.isclose(observed, row['best_val_pinball'], atol=1e-12, rtol=1e-12):
            raise ValueError('Validation history disagrees with record: ' + str(path))
        marker = json.loads((path / 'provenance.json').read_text())
        fingerprint = marker.pop('fingerprint')
        if hashlib.sha256(json.dumps(marker, sort_keys=True).encode()).hexdigest() != fingerprint:
            raise ValueError('Invalid provenance: ' + str(path))
        for key in ['candidate', 'dataset', 'horizon', 'seed', 'code_sha256', 'data_sha256']:
            if row.get(key) != marker.get(key):
                raise ValueError('Record identity differs from provenance: ' + key + ' ' + str(path))
        if row.get('candidate_id') != marker['candidate']['id']:
            raise ValueError('Record candidate ID differs from provenance: ' + str(path))
        if not protocol or row['code_sha256'] != protocol['source_hash']:
            raise ValueError('Record source differs from frozen protocol: ' + str(path))
        resolved = json.loads((path / 'resolved_config.json').read_text())
        config = dict(resolved['config'])
        config.pop('data_scaler', None)
        config.pop('feature_columns', None)
        if config != marker['config'] or config['paper']['stage'] != row['stage']:
            raise ValueError('Resolved configuration differs from provenance: ' + str(path))
        data_file = Path(config['data']['root_path']) / config['data']['files'][row['dataset']]
        if row['data_sha256'] != protocol['input_hashes'][str(data_file)] or cached_sha(data_file) != row['data_sha256']:
            raise ValueError('Dataset differs from frozen protocol: ' + str(path))
        if cached_sha(path / 'best_checkpoint.pt') != row['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from recorded digest: ' + str(path))
        if row['best_val_pinball'] > history.val_pinball.min() + 1e-8:
            raise ValueError('Recorded checkpoint is not validation-selected: ' + str(path))
        if row.get('test_evaluated'):
            if row['selection_sha256'] != cached_sha(ROOT / 'selection.json'):
                raise ValueError('Test record differs from frozen selection: ' + str(path))
            measured = json.loads((path / 'metrics.json').read_text())
            for key in ['QuantileLoss_standardized', 'PICP90', 'PINAW90', 'NormalizedWinkler90', 'CrossingRate']:
                if not np.isclose(row[key], measured[key], rtol=1e-12, atol=1e-12):
                    raise ValueError('Test metrics disagree: ' + str(path))
            if row['CrossingRate'] != 0:
                raise ValueError('Ordered candidate has quantile crossing')
            expected_windows = resolved['n_rows'] - resolved['split_bounds']['test'][0] - row['horizon'] + 1
            expected_elements = expected_windows * row['horizon'] * resolved['n_vars']
            if row.get('test_windows') != expected_windows or row.get('test_elements') != expected_elements:
                raise ValueError('Recorded test population is incomplete: ' + str(path))
        checked.append(str(path))
    return checked


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    validations = records('validation_complete')
    evaluated = records('complete')
    screen = [r for r in validations if r['stage'] == 'screen']
    confirmation = [r for r in validations if r['stage'] == 'confirm']
    selected = json.loads((ROOT / 'selection.json').read_text()) if (ROOT / 'selection.json').exists() else None
    figure_data = OUT / 'architecture_figure_data'
    figure_data.mkdir(exist_ok=True)
    curve_sources = []
    for record in confirmation:
        target = figure_data / (record['candidate_id'] + '_seed' + str(record['seed']) + '_history.csv')
        atomic_text(target, (Path(record['source_dir']) / 'training_history.csv').read_text())
        curve_sources.append(dict(candidate_id=record['candidate_id'], seed=record['seed'], history=str(target.relative_to(PAPER))))
    source_metadata = dict(screening='results/architecture_records.json: validation records with stage=screen',
                           confirmation_curves=curve_sources, selection=selected,
                           forecast_sample='first eligible test window, last saved variable, seed 2024')
    if selected:
        example = [r for r in evaluated if r['candidate_id'] == selected['candidate_id'] and
                   r['dataset'] == 'ETTh1' and r['horizon'] == 96 and r['seed'] == 2024]
        if example:
            target = figure_data / 'selected_ETTh1_H96_seed2024.npz'
            shutil.copyfile(Path(example[0]['source_dir']) / 'forecast_samples.npz', target)
            source_metadata['forecast_arrays'] = str(target.relative_to(PAPER))
    atomic_text(OUT / 'architecture_figure_sources.json', json.dumps(source_metadata, indent=2))
    aggregated = groups(evaluated)
    confirmed = {r['candidate_id']: r for r in groups(confirmation)}
    check = verify(validations + evaluated)
    atomic_text(OUT / 'architecture_records.json', json.dumps(dict(validation=validations, test=evaluated,
                selection=selected, aggregates=aggregated), indent=2))
    atomic_text(OUT / 'architecture_audit.json', json.dumps(dict(status='passed', checked_records=len(check),
                scope='Recorded validation histories, provenance fingerprints and source test metrics; independent neural re-evaluation is separate.'), indent=2))
    pd.DataFrame(aggregated).to_csv(OUT / 'architecture_aggregates.csv', index=False)
    pending = json.loads((ROOT / 'evaluation_queue.json').read_text()) if (ROOT / 'evaluation_queue.json').exists() else []
    status = dict(screen_completed=len(screen), screen_total=12, confirmation_completed=len(confirmation),
                  selection_frozen=selected is not None, test_completed=len(evaluated), test_total=len(pending),
                  complete_three_seed_test_groups=len(aggregated))
    status['worker_state'] = json.loads((ROOT / 'current_job.json').read_text()).get('status') if (ROOT / 'current_job.json').exists() else 'not started'
    if (ROOT / 'failure.json').exists():
        status['failure'] = json.loads((ROOT / 'failure.json').read_text())
    atomic_text(OUT / 'architecture_status.json', json.dumps(status, indent=2))
    lines = [r'\subsection{Validation-Selected Architecture Exploration}', r'\label{sec:architecture-search}',
        'We extend the original decoder study with a separate, prospective architecture search. '
        'All historical default-model results above are retained. The search uses ETTh1 at $H=96$ '
        'and selects exclusively by validation pinball; test scores do not influence candidate ranking. '
        'Twelve prespecified candidates compare the original three-layer Transformer, no encoder, '
        'residual per-token MLPs, dilated temporal CNNs, a CNN--Transformer combination, '
        'a one-layer Transformer, and linear/MLP/CNN/patch direct heads. '
        'MLP and CNN residual blocks use pre-layer normalization, GELU, dropout, and a final layer normalization. '
        'CNN blocks use kernel size 3, dilations 1/2/4 and a pointwise output projection; '
        'the hybrid uses two CNN blocks followed by one Transformer block. '
        'The narrow CNN/MLP variants jointly change width to 64, FFN width to 128, and learning rate to '
        '$3\\times10^{-4}$, so their comparison does not isolate those factors. '
        'The direct MLP maps the complete 336-step history through a 256-unit GELU hidden layer to $7H$ outputs. '
        'Every candidate retains training-only standardization and the ordered median/positive-gap output.',
        'Screening uses seed 2024 and at most eight epochs. The three candidates with lowest screening '
        'validation loss, together with the original model, are retrained with seeds 2024/2025/2026 '
        'under the original 30-epoch policy. The minimum mean confirmation validation loss determines '
        'a frozen candidate; candidate identifiers break exact ties. This finite search does not establish global optimality, '
        'and short screening can miss slower-converging configurations. Repeated validation use also introduces selection optimism.']
    rows = []
    for r in sorted(screen, key=lambda r: (r['best_val_pinball'], r['candidate_id'])):
        c = confirmed.get(r['candidate_id'])
        confirmation_text = pm(c, 'best_val_pinball') if c else '--'
        rows.append(' & '.join([LABELS[r['candidate_id']], str(r['n_params']),
                                 f"{r['best_val_pinball']:.5f}", confirmation_text]) + r' \\')
    if rows:
        lines.append(table('Architecture selection on ETTh1 validation data at $H=96$. Screening is a single seed with eight epochs; '
                           'confirmation entries require all three seeds and show mean $\\pm$ sample SD. Dashes denote unconfirmed groups, not zero loss. '
                           'Parameter counts are for $H=96$. No test score is used in this table.', 'tab:architecture-validation',
                           ['Candidate', 'Parameters', 'Screen val.', 'Confirm val.'], rows, 'lrrr'))
        lines.append(r'\begin{figure*}[t]\centering\includegraphics[width=.85\textwidth]{figures/architecture_screening.pdf}'
                     r'\caption{Completed eight-epoch validation screening runs. Bars rank validation loss; they are not test comparisons.}'
                     r'\Description{Horizontal bars compare completed architecture candidates on ETTh1 validation pinball.}\end{figure*}')
    lines.append(f"At this export, {len(screen)} of 12 screening runs and {len(confirmation)} confirmation runs have completed. "
                 f"There are {len(evaluated)} completed full-test runs and {len(aggregated)} complete three-seed test groups in the extension.")
    if selected:
        lines.append('The frozen selection is ' + LABELS[selected['candidate_id']] +
                     f", with mean confirmation validation pinball {selected['mean_val']:.5f}. "
                     'Its architecture is shown separately from the original HADT-NQ schematic in Figure~\\ref{fig:architecture-selected}. '
                     'Substituting standard modules is an empirical architecture choice, not a claim of a new component.')
        lines.append(r'\begin{figure*}[t]\centering\includegraphics[width=.98\textwidth]{figures/architecture_selected.pdf}'
                     r'\caption{Architecture frozen by ETTh1 validation selection. The original HADT-NQ remains the reference in Figure~\ref{fig:architecture}.}'
                     r'\Description{Computation graph of the validation-selected architecture.}\label{fig:architecture-selected}\end{figure*}')
        # Retrieve the selected candidate's own configuration, not the first shortlist member's.
        source = next(r for r in selected['confirmation_records'] if r['candidate_id'] == selected['candidate_id'])
        selected_config = json.loads((Path(source['source_dir']) / 'resolved_config.json').read_text())['config']
        selected_config.pop('data_scaler', None)
        selected_config.pop('feature_columns', None)
        atomic_text(Path('configs/selected_architecture.json'), json.dumps(dict(candidate=selected['candidate'], config=selected_config,
                    selection_source=str(ROOT / 'selection.json'), selection_uses_test=False), indent=2))
        lines.append('The prospective transfer queue compares the selected candidate, original HADT-NQ and Linear-NQ '
                     '(deduplicated when selection matches a control), with three seeds on ETTh1/ETTh2 at '
                     '$H\\in\\{96,192,336,720\\}$ and the same four horizons on ETTm1, ETTm2, Weather, Electricity, and Traffic. '
                     'Configuration selection is frozen across datasets and horizons. All use the same chronological ratio splits '
                     'and full-variable evaluation defined above. The selected model receives a validation-search budget '
                     'that the fixed controls do not receive; these comparisons therefore assess transfer after selection, '
                     'rather than equal hyperparameter-search effort. Pending jobs contribute no values or conclusions.')
    if confirmation:
        lines.append(r'\begin{figure*}[t]\centering\includegraphics[width=.98\textwidth]{figures/architecture_learning_curves.pdf}'
                     r'\caption{Validation learning curves of completed confirmation runs. Colors identify training seeds; '
                     r"checkpoints minimize each run's validation loss. These are not test curves.}"
                     r'\Description{Epoch-wise validation losses for each shortlisted candidate and completed seed.}\end{figure*}')
    if aggregated:
        frame = pd.DataFrame(aggregated)
        for horizon, subset in frame.groupby('horizon'):
            rows = []
            for r in subset.to_dict('records'):
                rows.append(' & '.join([r['dataset'], LABELS[r['candidate_id']], pm(r, 'QuantileLoss_standardized'),
                                         pm(r, 'PICP90', 100, 2), pm(r, 'NormalizedWinkler90', 1, 4)]) + r' \\')
            # Split blocks to prevent a completed broad study from overflowing a float.
            for part, start in enumerate(range(0, len(rows), 18)):
                lines.append(table(f'Completed architecture-study test groups, $H={horizon}$ (block {part + 1}). '
                                   'Mean $\\pm$ sample SD over three seeds; PICP is percent. '
                                   'Selection used validation only; no incomplete group is aggregated.',
                                   f'tab:architecture-test-{horizon}-{part}',
                                   ['Dataset', 'Candidate', 'Pinball', 'PICP90 (\\%)', 'Norm. Winkler90'],
                                   rows[start:start + 18], 'llrrr'))
        if selected:
            for (dataset, horizon), group in frame.groupby(['dataset', 'horizon']):
                index = {r['candidate_id']: r for r in group.to_dict('records')}
                winner = index.get(selected['candidate_id'])
                contrasts = []
                if winner is not None:
                    for comparator, label in [('transformer3', 'the original HADT-NQ'), ('linear', 'Linear-NQ')]:
                        reference = index.get(comparator)
                        if reference is not None and comparator != selected['candidate_id']:
                            change = 100 * (winner['QuantileLoss_standardized_mean'] / reference['QuantileLoss_standardized_mean'] - 1)
                            contrasts.append(f'{change:+.2f}\\% relative to {label}')
                if contrasts:
                    lines.append(f"On {dataset} at $H={horizon}$, the selected candidate changes mean test pinball by "
                                 + ' and '.join(contrasts) + ' (negative denotes improvement). '
                                 'These seed summaries do not establish statistical significance or universal superiority.')
    if selected and any(r['candidate_id'] == selected['candidate_id'] and r['dataset'] == 'ETTh1' and r['horizon'] == 96 and r['seed'] == 2024 for r in evaluated):
        lines.append(r'\begin{figure*}[t]\centering\includegraphics[width=.85\textwidth]{figures/architecture_forecast.pdf}'
                     r'\caption{Selected-model forecast for the first eligible ETTh1 test window, final variable, seed 2024. '
                     r'The sample was fixed independently of predictive quality; shading is the central 90\% interval.}'
                     r'\Description{Observed series, median and ordered prediction interval from the selected architecture.}\end{figure*}')
    plots(screen, confirmation, selected, evaluated)
    abstract = ''
    conclusion = ''
    method = ''
    if selected:
        abstract = ('A separate validation-only architecture search selects ' + LABELS[selected['candidate_id']] +
                    '; its completed transfer evaluations are reported separately.\n')
        conclusion = ('Within the explicitly tested architecture-search budget, ETTh1 validation selection favors ' +
                      LABELS[selected['candidate_id']] + '. The selected configuration is frozen across subsequent '
                      'datasets and horizons, and its completed test results are reported separately from the original study. '
                      'Broader optimality is not established.\n')
        c = selected['candidate']
        if c['encoder'] == 'mlp' and c['decoder'] == 'deformable':
            model_config = selected_config['model']
            method = (r'\paragraph{Validation-selected working configuration.} '
                      'The separate search in Section~\\ref{sec:architecture-search} freezes '
                      + LABELS[selected['candidate_id']] + ' as the working variant. '
                      f"It uses $D={model_config['d_model']}$ and three pre-normalized residual per-token FFNs "
                      f"with hidden width {model_config['d_ff']}, GELU, dropout 0.1, and final layer normalization. "
                      'Each block uses affine maps and computes '
                      '\\begin{equation}\\begin{aligned}'
                      'A&=\\operatorname{Drop}(\\operatorname{GELU}(W_1\\operatorname{LN}(Z))),\\\\'
                      "Z'&=Z+\\operatorname{Drop}(W_2 A)."
                      '\\end{aligned}\\end{equation} '
                      'These FFNs do not mix patch positions; temporal retrieval occurs in the horizon-conditioned '
                      'deformable decoder. Patch length, stride, eight decoder heads, $K=8$, offset bound, '
                      'RevIN and the ordered head are retained. The reference Transformer remains the comparator '
                      'in historical tables. Width and learning rate were selected jointly with encoder type, '
                      'so this configuration comparison does not isolate an MLP-only causal effect.\n')
    atomic_text(OUT / 'architecture_abstract.tex', abstract)
    atomic_text(OUT / 'architecture_conclusion.tex', conclusion)
    atomic_text(OUT / 'architecture_method.tex', method)
    text = '\n\n'.join(lines) + '\n'
    atomic_text(OUT / 'architecture_section.tex', text)
    audit_key = [selected['candidate_id'] if selected else None, len(aggregated), bool(evaluated)]
    audit_marker = ROOT / 'independent_audit_counts.json'
    previous_audit = json.loads(audit_marker.read_text()) if audit_marker.exists() else None
    if (selected or evaluated) and previous_audit != audit_key:
        subprocess.run([sys.executable, 'scripts/validate_architecture_study.py'], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        atomic_text(audit_marker, json.dumps(audit_key))
    # Saved forecasts are rendered on CPU; compatible prior images are reused.
    from scripts.plot_run_forecasts import render_completed
    status['prediction_plots_completed'] = render_completed(evaluated, workers=1)
    atomic_text(OUT / 'architecture_status.json', json.dumps(status, indent=2))
    # Rebuild when the completed evidence or selection changes. Build failures remain visible.
    tex_inputs = [PAPER / 'main.tex', *sorted((PAPER / 'sections').glob('*.tex')), *sorted(OUT.glob('*.tex'))]
    stamp = hashlib.sha256(''.join(path.read_text() for path in tex_inputs).encode()).hexdigest()
    marker = ROOT / 'paper_build.sha256'
    if not marker.exists() or marker.read_text() != stamp:
        with open(ROOT / 'paper_build.log', 'w') as log:
            subprocess.run(['bash', 'scripts/build_paper.sh'], stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, TECTONIC_OFFLINE='1'))
        atomic_text(marker, stamp)
    # Keep this extension reproducible without altering the historical delivery archive.
    source_paths = []
    for directory in ['models', 'data_provider', 'engine', 'metrics', 'losses', 'utils', 'experiments', 'configs', 'tests']:
        source_paths.extend(p for p in Path(directory).glob('*') if p.suffix in ('.py', '.yaml', '.json'))
    source_paths += [Path('scripts') / name for name in ['run_architecture_study.py', 'export_architecture_study.py',
                     'validate_architecture_study.py', 'plot_run_forecasts.py', 'run_paper_experiments.py', 'build_paper.sh']]
    source_paths += [Path('requirements.txt'), Path('docs/ARCHITECTURE_STUDY.md')]
    archive = PAPER / 'supplementary/architecture_source.zip'
    temporary = archive.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in source_paths:
            if path.exists():
                zipped.write(path, path.as_posix())
        for name in ['protocol.json', 'selection.json', 'evaluation_queue.json']:
            path = ROOT / name
            if path.exists():
                zipped.write(path, path.as_posix())
    temporary.replace(archive)
    print('ARCHITECTURE EXPORT', json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
