import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from configs import config
from coreset import compute_dataset_curvature
from dataset import STATS, get_class_to_indices, get_train_dataset_no_aug
from models import get_model

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def unnormalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(STATS[config.DATASET]['mean']).view(3, 1, 1)
    std = torch.tensor(STATS[config.DATASET]['std']).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def visualize(optimizer: str, n_per_class: int = 10) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = os.path.join(config.RESULTS_ROOT, optimizer, f'seed_{config.TUNE_SEED}', 'selection.pt')

    dataset = get_train_dataset_no_aug(config.DATASET, config.DATA_ROOT)
    model = get_model(config.ARCH, config.DATASET).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    gamma = compute_dataset_curvature(
        model, dataset, nn.CrossEntropyLoss(reduction='none'),
        h=config.H, batch_size=256, device=device,
    )
    print(f"gamma  min={gamma.min():.3f}  max={gamma.max():.3f}  mean={gamma.mean():.3f}")

    class_to_idx = get_class_to_indices(dataset)
    out_dir = os.path.join(config.RESULTS_ROOT, optimizer)

    for title, reverse in [('Low curvature (clean)', False), ('High curvature (hard)', True)]:
        fig = plt.figure(figsize=(n_per_class * 1.2, 11))
        gs = gridspec.GridSpec(10, n_per_class, hspace=0.1, wspace=0.05)

        for row, label in enumerate(sorted(class_to_idx)):
            indices = torch.tensor(class_to_idx[label])
            order = gamma[indices].argsort(descending=reverse)
            for col, idx in enumerate(indices[order[:n_per_class]].tolist()):
                img, _ = dataset[idx]
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(unnormalize(img).permute(1, 2, 0).numpy())
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(CIFAR10_CLASSES[label], fontsize=8, rotation=0,
                                  labelpad=40, va='center')

        fig.suptitle(f'{title} - {optimizer}', fontsize=13, y=1.01)
        path = os.path.join(out_dir, f"{'high' if reverse else 'low'}_curvature_samples.png")
        plt.savefig(path, dpi=120, bbox_inches='tight')
        print(f"Saved {path}")
        plt.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--optimizer', default='sgd', choices=list(config.OPTIMIZERS))
    args = p.parse_args()
    visualize(args.optimizer)