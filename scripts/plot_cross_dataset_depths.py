"""Compare within-dataset relative depth effects without mixing absolute scales."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path('outputs/Traffic_depth_comparison')
    frame = pd.read_csv(root / 'cross_dataset_comparison.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    output = pd.DataFrame({'layers': frame.layers})
    for ax, split in zip(axes, ['val', 'test']):
        for dataset in ['ETTh1', 'Traffic']:
            values = frame[f'{split}_pinball_{dataset}'].to_numpy()
            baseline = frame.loc[frame.layers == 0, f'{split}_pinball_{dataset}'].iloc[0]
            relative = (values / baseline - 1) * 100
            output[f'{split}_change_percent_{dataset}'] = relative
            ax.plot(frame.layers.to_numpy(), relative, 'o-', label=dataset)
        ax.axhline(0, color='0.4', linestyle='--')
        ax.set(title=f'{"Validation" if split == "val" else "Test"} pinball relative to 0 layers',
               xlabel='Encoder layers (0 = bypass)', ylabel='Loss change (%) | negative = improvement')
        ax.set_xticks([0, 1, 2, 3])
        ax.legend()
        ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(root / f'cross_dataset_relative_loss.{ext}', dpi=160, bbox_inches='tight')
    plt.close(fig)
    output.to_csv(root / 'cross_dataset_relative_loss.csv', index=False)
    report = root / 'RESULTS.md'
    caption = '\n![相对0层的损失变化：负数表示改善](cross_dataset_relative_loss.png)\n'
    if report.exists() and 'cross_dataset_relative_loss.png' not in report.read_text():
        with report.open('a', encoding='utf-8') as handle:
            handle.write(caption)
    print(output.to_string(index=False))


if __name__ == '__main__':
    main()
