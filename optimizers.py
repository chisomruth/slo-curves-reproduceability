import torch
import torch.nn as nn
from typing import List

from configs import config


def _newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for hidden weights. Conv kernels are flattened to (out, in*k*k)."""
    def __init__(self, params, lr: float, momentum: float = 0.95, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(p.grad)
                buf = state['buf']
                buf.mul_(group['momentum']).add_(p.grad)
                update = p.grad.add(buf, alpha=group['momentum'])       # Nesterov
                update = _newton_schulz(update.reshape(p.size(0), -1))
                update = update * max(1.0, update.size(0) / update.size(1)) ** 0.5
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update.reshape(p.shape), alpha=-group['lr'])


def build_optimizer(name: str, model: nn.Module, lr: float) -> List[torch.optim.Optimizer]:
    """Returns a list of optimizers (Muon needs two: Muon + AdamW for the rest)."""
    cfg = config.OPTIMIZERS[name]
    wd = cfg['weight_decay']
    params = model.parameters()

    if name == 'sgd':
        return [torch.optim.SGD(params, lr=lr, momentum=config.MOMENTUM,
                                nesterov=True, weight_decay=wd)]
    if name == 'adamw':
        return [torch.optim.AdamW(params, lr=lr, weight_decay=wd)]
    if name == 'muon':
        # Muon: hidden weights. AdamW: first conv, final fc, BN and biases.
        weights = [p for p in model.parameters() if p.ndim >= 2]
        special = {id(weights[0]), id(weights[-1])}
        muon_params = [p for p in weights if id(p) not in special]
        adam_params = [p for p in model.parameters() if p.ndim < 2 or id(p) in special]
        return [
            Muon(muon_params, lr=lr, weight_decay=wd),
            torch.optim.AdamW(adam_params, lr=cfg['aux_lr'], weight_decay=wd),
        ]
    if name == 'soap':
        from soap import SOAP     # soap.py from github.com/nikhilvyas/SOAP
        return [SOAP(params, lr=lr, weight_decay=wd,
                     precondition_frequency=cfg['precondition_frequency'])]
    if name == 'shampoo':
        import torch_optimizer
        return [torch_optimizer.Shampoo(params, lr=lr, momentum=config.MOMENTUM,
                                        weight_decay=wd, update_freq=cfg['update_freq'])]
    raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(config.OPTIMIZERS)}")