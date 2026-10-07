# SLo-Curves across optimizers

Reproduces SLo-Curves (Garg & Roy, CVPR 2023) and tests whether the result holds for
SGD, AdamW, Muon, SOAP and Shampoo. CIFAR-10, ResNet18, 1000-sample coreset (100/class).

For each optimizer (one fixed LR, same for selection model and coreset training):
1. Train the selection model (40 epochs) and pick the 100 lowest-curvature samples per class.
2. Tune λ on a held-out 1k validation split (seed 0 only).
3. Run seeds 0-4 on the test set with two conditions: `random` and `slo_best` (SLo + best λ).

## Setup (Lightning AI Studio)
```bash
pip install -r requirements.txt
curl -O https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py
python dataset.py                      # download CIFAR-10 once
python run.py --optimizer sgd --smoke  # tiny end-to-end check
```

## Run
```bash
python run.py --optimizer sgd
python run.py --optimizer all          # sequential
python run.py --optimizer adamw --num_seeds 3
CUDA_VISIBLE_DEVICES=1 python run.py --optimizer muon   # one optimizer per GPU
python plot.py
python viz_curv.py --optimizer sgd
```
Re-running the same command resumes from where it stopped.

## Results layout
```
results/<optimizer>/
  config.json   tuning.json   runs.json   summary.json
  seed_<k>/coreset_slo.json  coreset_random.json  gamma.pt  (selection.pt for seed 0)
```

## Notes
- The LR schedule keeps the paper's shape (decay at 81/164 and 121/164 of training), scaled to the
  40-epoch selection run.
- The regulariser runs the perturbed forward pass in train mode, so BatchNorm running stats
  also see perturbed inputs.