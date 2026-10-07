import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

STATS = {
    'cifar10':  {'mean': (0.4914, 0.4822, 0.4465),
                 'std':  (0.2023, 0.1994, 0.2010)},
    'cifar100': {'mean': (0.5071, 0.4867, 0.4408),
                 'std':  (0.2675, 0.2565, 0.2761)},
}

_DATASET_CLS = {
    'cifar10':  torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
}


class IndexedDataset(Dataset):
    """Wraps a dataset so it also returns the position index as a third element."""
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        x, y = self.dataset[idx]
        return x, y, idx


def get_transforms(dataset: str, train: bool) -> T.Compose:
    mean = STATS[dataset]['mean']
    std  = STATS[dataset]['std']
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def get_train_dataset(dataset: str, root: str) -> Dataset:
    """Training split with augmentation (used to train models)."""
    return _DATASET_CLS[dataset](
        root=root, train=True,
        transform=get_transforms(dataset, train=True), download=True,
    )


def get_train_dataset_no_aug(dataset: str, root: str) -> Dataset:
    """Training split without augmentation (used for curvature and validation)."""
    return _DATASET_CLS[dataset](
        root=root, train=True,
        transform=get_transforms(dataset, train=False), download=True,
    )


def get_test_dataset(dataset: str, root: str) -> Dataset:
    return _DATASET_CLS[dataset](
        root=root, train=False,
        transform=get_transforms(dataset, train=False), download=True,
    )


def get_class_to_indices(dataset: Dataset) -> Dict[int, List[int]]:
    class_to_idx: Dict[int, List[int]] = {}
    for idx, label in enumerate(dataset.targets):
        class_to_idx.setdefault(int(label), []).append(idx)
    return class_to_idx


def split_train_val(targets, val_per_class: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced split of the training set into (pool_idx, val_idx)."""
    gen = torch.Generator().manual_seed(seed)
    labels = torch.tensor(targets)
    pool, val = [], []
    for c in labels.unique(sorted=True):
        idx = (labels == c).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=gen)]
        val.append(idx[:val_per_class])
        pool.append(idx[val_per_class:])
    return torch.cat(pool).sort().values, torch.cat(val).sort().values


if __name__ == '__main__':
    from configs import config
    get_train_dataset(config.DATASET, config.DATA_ROOT)
    get_test_dataset(config.DATASET, config.DATA_ROOT)
    print(f"{config.DATASET} ready in {config.DATA_ROOT}")