# SLo-Curves across optimizers

Reproduces SLo-Curves (Garg & Roy, CVPR 2023) and tests whether the result holds for
SGD, AdamW, Muon, SOAP and Shampoo. Using CIFAR-10 on ResNet18.

For each optimizer:
1. Train the selection model (40 epochs) and pick the 100 lowest-curvature samples per class.
2. Tune λ on a held-out 1k validation split.
3. Run seeds 0-4 on the test set with two conditions: `random` and `slo_best` (SLo + best λ).

## Setup (Lightning AI Studio)
```bash
pip install -r requirements.txt
curl -O https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py
python dataset.py                      # download CIFAR-10 once
```

## Run
```bash
python run.py --optimizer all          # sequential
python plot.py
python viz_curv.py --optimizer all 
```

## Notes
- The LR schedule keeps the paper's shape (decay at 81/164 and 121/164 of training), scaled to the
  40-epoch selection run.
- The regulariser runs the perturbed forward pass in train mode, so BatchNorm running stats
  also see perturbed inputs.
