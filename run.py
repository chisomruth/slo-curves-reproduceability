import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from configs import config
from coreset import compute_dataset_curvature, select_lowest_curvature, select_random
from dataset import (
    get_test_dataset,
    get_train_dataset,
    get_train_dataset_no_aug,
    split_train_val,
)
from models import get_model
from train import evaluate, train


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path: str, obj) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)      


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0)


@dataclass
class Data:
    train_aug: Dataset
    train_noaug: Dataset
    pool_idx: torch.Tensor        
    pool_labels: torch.Tensor
    val_loader: DataLoader        # held-out 1k split, used for tuning only
    test_loader: DataLoader   

def build_data() -> Data:
    train_aug = get_train_dataset(config.DATASET, config.DATA_ROOT)
    train_noaug = get_train_dataset_no_aug(config.DATASET, config.DATA_ROOT)
    test_ds = get_test_dataset(config.DATASET, config.DATA_ROOT)

    pool_idx, val_idx = split_train_val(train_noaug.targets, config.VAL_PER_CLASS, config.SPLIT_SEED)
    pool_labels = torch.tensor(train_noaug.targets)[pool_idx]
    return Data(
        train_aug=train_aug,
        train_noaug=train_noaug,
        pool_idx=pool_idx,
        pool_labels=pool_labels,
        val_loader=make_loader(Subset(train_noaug, val_idx.tolist()), 256, False, 2),
        test_loader=make_loader(test_ds, 256, False, 2),
    )



def train_selection_model(opt: str, lr: float, data: Data, device: torch.device) -> nn.Module:
    model = get_model(config.ARCH, config.DATASET).to(device)
    loader = make_loader(Subset(data.train_aug, data.pool_idx.tolist()),
                         config.BATCH_SIZE, True, 4)
    model, _ = train(model, loader, None, opt, lr, config.SELECTION_EPOCHS, 0.0, device)
    return model


def get_coresets(opt: str, lr: float, seed: int, data: Data, device: torch.device,
                 out_dir: str) -> Tuple[List[int], List[int]]:
    seed_dir = os.path.join(out_dir, f'seed_{seed}')
    os.makedirs(seed_dir, exist_ok=True)
    slo_path = os.path.join(seed_dir, 'coreset_slo.json')
    rnd_path = os.path.join(seed_dir, 'coreset_random.json')

    if not os.path.exists(rnd_path):
        save_json(rnd_path, select_random(data.pool_labels, data.pool_idx, config.SPC, seed))

    if not os.path.exists(slo_path):
        t0 = time.time()
        set_seed(seed)
        model = train_selection_model(opt, lr, data, device)
        val_acc = evaluate(model, data.val_loader, device)
        gamma = compute_dataset_curvature(
            model, Subset(data.train_noaug, data.pool_idx.tolist()),
            nn.CrossEntropyLoss(reduction='none'), config.H, 256, device,
        )
        print(f"  [seed {seed}] selection model val acc {val_acc:.2f}% | "
              f"gamma min {gamma.min():.4f} mean {gamma.mean():.4f} max {gamma.max():.4f} | "
              f"{(time.time() - t0) / 60:.1f} min")
        torch.save(gamma, os.path.join(seed_dir, 'gamma.pt'))
        if seed == config.TUNE_SEED:
            torch.save(model.state_dict(), os.path.join(seed_dir, 'selection.pt'))
        save_json(slo_path, select_lowest_curvature(gamma, data.pool_labels, data.pool_idx, config.SPC))

    return load_json(slo_path, None), load_json(rnd_path, None)


def train_on_coreset(opt: str, lr: float, lam: float, indices: List[int], eval_loader: DataLoader,
                     data: Data, device: torch.device, seed: int) -> Tuple[float, list]:
    set_seed(seed)     
    loader = make_loader(Subset(data.train_aug, indices),
                         min(config.BATCH_SIZE, len(indices)), True, 2)
    model = get_model(config.ARCH, config.DATASET).to(device)
    model, curve = train(model, loader, eval_loader, opt, lr, config.EPOCHS, lam, device)
    return curve[-1][1], curve


def tune_lambda(opt: str, lr: float, data: Data, device: torch.device, out_dir: str) -> float:
    path = os.path.join(out_dir, 'tuning.json')
    tuning = load_json(path, {'lambda_sweep': {}})
    seed = config.TUNE_SEED

    slo_idx, _ = get_coresets(opt, lr, seed, data, device, out_dir)
    for lam in config.LAMBDA_SEARCH:
        if str(lam) in tuning['lambda_sweep']:
            continue
        acc, _ = train_on_coreset(opt, lr, lam, slo_idx, data.val_loader, data, device, seed)
        tuning['lambda_sweep'][str(lam)] = acc
        save_json(path, tuning)
        print(f"  [tune] lambda={lam:<5} val acc {acc:.2f}%")

    best_lam = float(max(tuning['lambda_sweep'], key=tuning['lambda_sweep'].get))
    tuning['lr'] = lr
    tuning['best_lambda'] = best_lam
    save_json(path, tuning)
    print(f"  [tune] best lambda={best_lam}")
    return best_lam



def summarize(runs: Dict[str, dict]) -> dict:
    by_cond: Dict[str, Dict[int, float]] = {}
    for r in runs.values():
        by_cond.setdefault(r['condition'], {})[r['seed']] = r['test_acc']

    summary = {}
    for cond, accs in by_cond.items():
        vals = list(accs.values())
        summary[cond] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                         'n': len(vals), 'accs': vals}
        if cond != 'random' and 'random' in by_cond:
            gains = [accs[s] - by_cond['random'][s] for s in accs if s in by_cond['random']]
            summary[cond]['gain_vs_random'] = float(np.mean(gains)) if gains else None
    return summary


def final_runs(opt: str, lr: float, lam: float, num_seeds: int, data: Data,
               device: torch.device, out_dir: str) -> None:
    runs_path = os.path.join(out_dir, 'runs.json')
    runs = load_json(runs_path, {})

    for seed in range(num_seeds):
        slo_idx, rnd_idx = get_coresets(opt, lr, seed, data, device, out_dir)
        conditions = [('random', 0.0, rnd_idx), ('slo_best', lam, slo_idx)]

        for cond, l, idx in conditions:
            key = f'seed{seed}/{cond}'
            if key in runs:
                continue
            t0 = time.time()
            acc, curve = train_on_coreset(opt, lr, l, idx, data.test_loader, data, device, seed)
            runs[key] = {'seed': seed, 'condition': cond, 'lambda': l, 'lr': lr,
                         'test_acc': acc, 'curve': curve}
            save_json(runs_path, runs)
            save_json(os.path.join(out_dir, 'summary.json'), summarize(runs))
            print(f"  [seed {seed}] {cond:<9} test acc {acc:.2f}%  ({(time.time() - t0) / 60:.1f} min)")


def run_optimizer(opt: str, num_seeds: int, data: Data, device: torch.device) -> None:
    out_dir = os.path.join(config.RESULTS_ROOT, opt)
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, 'config.json'),
              {k: getattr(config, k) for k in dir(config) if k.isupper()})

    print(f"\n{'=' * 60}\nOptimizer: {opt}\n{'=' * 60}")
    lr = config.OPTIMIZERS[opt]['lr']
    lam = tune_lambda(opt, lr, data, device, out_dir)
    final_runs(opt, lr, lam, num_seeds, data, device, out_dir)

    print(f"\n  Summary ({opt}):")
    for cond, s in load_json(os.path.join(out_dir, 'summary.json'), {}).items():
        print(f"    {cond:<9} {s['mean']:.2f}% ± {s['std']:.2f}%  (n={s['n']})")


def apply_smoke() -> None:
    config.EPOCHS = 2
    config.SELECTION_EPOCHS = 1
    config.EVAL_EVERY = 1
    config.LAMBDA_SEARCH = [1]
    config.RESULTS_ROOT = './results_smoke'


def main() -> None:
    p = argparse.ArgumentParser(description='SLo-Curves across optimizers')
    p.add_argument('--optimizer', default='sgd', choices=list(config.OPTIMIZERS) + ['all'])
    p.add_argument('--num_seeds', type=int, default=config.NUM_SEEDS)
    p.add_argument('--smoke', action='store_true', help='tiny end-to-end test run')
    args = p.parse_args()

    if args.smoke:
        apply_smoke()
        args.num_seeds = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    data = build_data()
    n_classes = len(data.pool_labels.unique())
    print(f"device={device}  optimizer={args.optimizer}  seeds={args.num_seeds}  "
          f"coreset={config.SPC * n_classes} samples")
    names = list(config.OPTIMIZERS) if args.optimizer == 'all' else [args.optimizer]
    for name in names:
        run_optimizer(name, args.num_seeds, data, device)


if __name__ == '__main__':
    main()