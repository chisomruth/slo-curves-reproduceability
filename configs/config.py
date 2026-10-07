DATASET = 'cifar10'
ARCH = 'resnet18'
DATA_ROOT = './data'
RESULTS_ROOT = './results'

# Coreset / validation sizes (CIFAR-10: 100 per class -> 1000 samples)
SPC = 100
VAL_PER_CLASS = 100 
SPLIT_SEED = 1234      

EPOCHS = 164
SELECTION_EPOCHS = 40
BATCH_SIZE = 128
LR_DECAY_FRACTIONS = [81 / 164, 121 / 164]  
LR_DECAY_FACTOR = 0.1
MOMENTUM = 0.9
EVAL_EVERY = 10

H = 3.0
LAMBDA_SEARCH = [0, 0.5, 1, 5, 10, 20, 50]

NUM_SEEDS = 5
TUNE_SEED = 0

OPTIMIZERS = {
    'sgd':     {'lr': 0.1,   'weight_decay': 5e-4},      # used by the paper
    'adamw':   {'lr': 1e-3,  'weight_decay': 5e-2},
    'muon':    {'lr': 0.02,  'weight_decay': 1e-2, 'aux_lr': 1e-3},
    'soap':    {'lr': 3e-3,  'weight_decay': 5e-2, 'precondition_frequency': 20},
    'shampoo': {'lr': 0.03,  'weight_decay': 5e-4, 'update_freq': 100},
}