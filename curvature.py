import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


def _per_sample_input_gradient(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    criterion_none: nn.Module,
    create_graph: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Returns (dL/dx per sample, per-sample loss)."""
    x_req = x.detach().requires_grad_(True)
    per_sample_loss = criterion_none(model(x_req), y)
    grad = torch.autograd.grad(
        per_sample_loss.sum(), x_req, create_graph=create_graph
    )[0]
    return grad, per_sample_loss


def _adversarial_direction(grad: Tensor) -> Tensor:
    """z = sign(g) / ||sign(g)||  (paper, Sec. 3.1)."""
    B = grad.shape[0]
    sign_g = grad.sign()
    norm = sign_g.view(B, -1).norm(dim=1).clamp(min=1e-8)
    return sign_g / norm.view(B, 1, 1, 1)


@torch.no_grad()
def compute_gamma_batch(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    criterion_none: nn.Module,
    h: float,
) -> Tensor:
    """gamma(x) = ||grad L(x + h z) - grad L(x)||, no graph kept (selection)."""
    model.eval()
    with torch.enable_grad():
        grad1, _ = _per_sample_input_gradient(model, x, y, criterion_none)
        z = _adversarial_direction(grad1.detach())
        x_hat = x.detach() + h * z
        grad2, _ = _per_sample_input_gradient(model, x_hat, y, criterion_none)

    B = x.shape[0]
    return (grad2.detach() - grad1.detach()).view(B, -1).norm(dim=1)


def compute_gamma_for_training(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    criterion_none: nn.Module,
    h: float,
) -> Tuple[Tensor, Tensor]:
    grad1, per_sample_loss = _per_sample_input_gradient(
        model, x, y, criterion_none, create_graph=True
    )
    z = _adversarial_direction(grad1.detach())
    x_hat = x.detach() + h * z
    grad2, _ = _per_sample_input_gradient(
        model, x_hat, y, criterion_none, create_graph=True
    )

    B = x.shape[0]
    gamma = (grad2 - grad1).view(B, -1).norm(dim=1)
    return gamma, per_sample_loss.mean()