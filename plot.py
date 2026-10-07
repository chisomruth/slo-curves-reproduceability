import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = [
    ('random',   'Random',              '#aaaaaa'),
    ('slo_best', 'SLo + best λ',        '#e6194b'),
]


def load_results(results_dir: str) -> dict:
    results = {}
    for opt in sorted(os.listdir(results_dir)):
        folder = os.path.join(results_dir, opt)
        if not os.path.isdir(folder):
            continue
        entry = {}
        for name in ('summary', 'tuning'):
            path = os.path.join(folder, f'{name}.json')
            if os.path.exists(path):
                with open(path) as f:
                    entry[name] = json.load(f)
        results[opt] = entry
    return results


def plot_accuracy(results: dict, out_dir: str) -> None:
    opts = [o for o in results if 'summary' in results[o]]
    x = np.arange(len(opts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (cond, label, color) in enumerate(CONDITIONS):
        means = [results[o]['summary'].get(cond, {}).get('mean', np.nan) for o in opts]
        stds = [results[o]['summary'].get(cond, {}).get('std', 0.0) for o in opts]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
               label=label, color=color, edgecolor='k', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(opts)
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('ResNet18, CIFAR-10, 1000-sample coreset')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(out_dir, 'accuracy_per_optimizer.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def plot_lambda_sweep(results: dict, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for opt, entry in results.items():
        sweep = entry.get('tuning', {}).get('lambda_sweep')
        if not sweep:
            continue
        lams = [float(k) for k in sweep]
        ax.plot(range(len(lams)), list(sweep.values()), marker='o', label=opt)
        ax.set_xticks(range(len(lams)))
        ax.set_xticklabels([f'{l:g}' for l in lams])
    ax.set_xlabel('λ')
    ax.set_ylabel('Validation accuracy (%)')
    ax.set_title('λ sweep on held-out validation (seed 0)')
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, 'lambda_sweep.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


def print_table(results: dict) -> None:
    print(f"\n{'optimizer':<9} {'random':>14} {'slo best λ':>14} {'gain':>8} {'λ':>6}")
    for opt, entry in results.items():
        s = entry.get('summary', {})
        rnd, slo = s.get('random'), s.get('slo_best')
        fmt = lambda c: f"{c['mean']:.2f}±{c['std']:.2f}" if c else '-'
        gain = slo.get('gain_vs_random') if slo else None
        lam = entry.get('tuning', {}).get('best_lambda', '-')
        print(f"{opt:<9} {fmt(rnd):>14} {fmt(slo):>14} "
              f"{(f'{gain:+.2f}' if gain is not None else '-'):>8} {lam:>6}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Plot SLo-Curves results per optimizer')
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--out_dir', default='./plots')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results = load_results(args.results_dir)
    print_table(results)
    plot_accuracy(results, args.out_dir)
    plot_lambda_sweep(results, args.out_dir)