import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Optional, Tuple

from configs import config
from curvature import compute_gamma_for_training
from optimizers import build_optimizer


def build_schedulers(optimizers: List[torch.optim.Optimizer], num_epochs: int):
    milestones = [max(1, round(f * num_epochs)) for f in config.LR_DECAY_FRACTIONS]
    return [
        torch.optim.lr_scheduler.MultiStepLR(o, milestones=milestones, gamma=config.LR_DECAY_FACTOR)
        for o in optimizers
    ]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizers: List[torch.optim.Optimizer],
    criterion: nn.Module,
    criterion_none: nn.Module,
    lambda_reg: float,
    h: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        for o in optimizers:
            o.zero_grad(set_to_none=True)

        if lambda_reg == 0:
            loss = criterion(model(x), y)
        else:
            gamma, loss_ce = compute_gamma_for_training(model, x, y, criterion_none, h)
            loss = loss_ce + lambda_reg * gamma.mean()

        loss.backward()
        for o in optimizers:
            o.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy (%)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader],
    optimizer_name: str,
    lr: float,
    num_epochs: int,
    lambda_reg: float,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[nn.Module, List[Tuple[int, float]]]:
    criterion = nn.CrossEntropyLoss()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    optimizers = build_optimizer(optimizer_name, model, lr)
    schedulers = build_schedulers(optimizers, num_epochs)

    curve: List[Tuple[int, float]] = []
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizers, criterion, criterion_none,
                           lambda_reg, config.H, device)
        for s in schedulers:
            s.step()

        diverged = not math.isfinite(loss)
        if eval_loader is not None and (diverged or epoch % config.EVAL_EVERY == 0 or epoch == num_epochs):
            acc = evaluate(model, eval_loader, device)
            curve.append((epoch, acc))
            if verbose:
                print(f"    epoch {epoch:3d}/{num_epochs} | loss {loss:.4f} | acc {acc:.2f}%")
        if diverged:
            print(f"    loss diverged at epoch {epoch}; stopping this run")
            break

    return model, curve