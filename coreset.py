import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing import List

from curvature import compute_gamma_batch
from dataset import IndexedDataset


def compute_dataset_curvature(
    model: nn.Module,
    dataset: Dataset,
    criterion_none: nn.Module,
    h: float,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """gamma for every sample in `dataset` (aligned with dataset positions)."""
    loader = DataLoader(
        IndexedDataset(dataset), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    gamma = torch.zeros(len(dataset))
    model.eval()
    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        gamma[idx] = compute_gamma_batch(model, x, y, criterion_none, h).cpu()
    return gamma


def select_lowest_curvature(
    gamma: Tensor, labels: Tensor, pool_idx: Tensor, spc: int,
) -> List[int]:
    selected: List[int] = []
    for c in labels.unique(sorted=True):
        pos = (labels == c).nonzero().squeeze(1)
        chosen = pos[gamma[pos].argsort()[:spc]]
        selected.extend(pool_idx[chosen].tolist())
    return selected


def select_random(
    labels: Tensor, pool_idx: Tensor, spc: int, seed: int,
) -> List[int]:
    gen = torch.Generator().manual_seed(seed)
    selected: List[int] = []
    for c in labels.unique(sorted=True):
        pos = (labels == c).nonzero().squeeze(1)
        chosen = pos[torch.randperm(len(pos), generator=gen)[:spc]]
        selected.extend(pool_idx[chosen].tolist())
    return selected