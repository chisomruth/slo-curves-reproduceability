import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch

from configs import config
from coreset import select_lowest_curvature
from dataset import get_train_dataset_no_aug, split_train_val


def visualize(optimizer: str, seed: int, n_per_class: int = 10) -> None:
    """Uses results/<optimizer>/seed_<seed>/gamma.pt saved by run.py (no GPU or model needed).
    Saves low/high curvature sample grids, a gamma histogram and the chosen indices
    into results/<optimizer>/."""
    folder = os.path.join(config.RESULTS_ROOT, optimizer)
    gamma_path = os.path.join(folder, f'seed_{seed}', 'gamma.pt')
    if not os.path.exists(gamma_path):
        print(f"[{optimizer}] {gamma_path} not found, skipping")
        return

    gamma = torch.load(gamma_path)
    dataset = get_train_dataset_no_aug(config.DATASET, config.DATA_ROOT)
    pool_idx, _ = split_train_val(dataset.targets, config.VAL_PER_CLASS, config.SPLIT_SEED)
    labels = torch.tensor(dataset.targets)[pool_idx]

    # lowest gamma first; negating gamma gives the highest-gamma samples
    picks = {
        'low':  select_lowest_curvature(gamma, labels, pool_idx, n_per_class),
        'high': select_lowest_curvature(-gamma, labels, pool_idx, n_per_class),
    }
    with open(os.path.join(folder, f'curvature_samples_seed{seed}.json'), 'w') as f:
        json.dump(picks, f)

    # gamma histogram (paper Fig. 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gamma.numpy(), bins=200, color='b')
    ax.set_xlabel('γ (curvature estimate)')
    ax.set_ylabel('Number of training samples')
    ax.set_title(f'{optimizer}, seed {seed}')
    fig.savefig(os.path.join(folder, f'gamma_hist_seed{seed}.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    titles = {'low': 'Lowest curvature (clean)', 'high': 'Highest curvature (hard)'}
    n_classes = len(dataset.classes)
    for key, indices in picks.items():
        fig = plt.figure(figsize=(n_per_class * 1.2, n_classes * 1.1))
        gs = gridspec.GridSpec(n_classes, n_per_class, hspace=0.1, wspace=0.05)
        for i, idx in enumerate(indices):
            row, col = divmod(i, n_per_class)
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(dataset.data[idx])
            ax.axis('off')
            if col == 0:
                ax.set_title(dataset.classes[row], fontsize=8, loc='left')
        fig.suptitle(f'{titles[key]} - {optimizer}, seed {seed}', fontsize=13, y=1.01)
        path = os.path.join(folder, f'{key}_curvature_samples_seed{seed}.png')
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"[{optimizer}] saved {path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--optimizer', default='sgd', choices=list(config.OPTIMIZERS) + ['all'])
    p.add_argument('--seed', type=int, default=config.TUNE_SEED)
    p.add_argument('--n_per_class', type=int, default=10)
    args = p.parse_args()

    names = list(config.OPTIMIZERS) if args.optimizer == 'all' else [args.optimizer]
    for name in names:
        visualize(name, args.seed, args.n_per_class)