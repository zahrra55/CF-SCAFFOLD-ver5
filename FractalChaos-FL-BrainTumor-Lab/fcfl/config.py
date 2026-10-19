"""Paths and experimental settings.

Paths can be changed with the environment variables FCFL_INPUT and FCFL_OUTPUT.
Setting FCFL_SMOKE_TEST=1 shrinks every setting so that all experiments run in a
few minutes; this checks the installation and does not reproduce the results.
"""

import os

import torch

INPUT_ROOT = os.environ.get("FCFL_INPUT", "/kaggle/input")
OUTPUT_DIR = os.environ.get("FCFL_OUTPUT", "/kaggle/working")
SMOKE_TEST = os.environ.get("FCFL_SMOKE_TEST") == "1"

FIGSHARE_GLOB = f"{INPUT_ROOT}/**/*.mat"
BRATS_GLOB = f"{INPUT_ROOT}/**/*t1ce.nii*"

IMG_SIZE = 128
N_FOLDS = 5
TEST_TRAIN_RATIO = 1.0 / (N_FOLDS - 1)
TEST_SPLIT_SEED = 42
BRATS_SLICES, BRATS_MIN_AREA = 5, 100

N_REP = 2 if SMOKE_TEST else 10
NESTED_REP = 1 if SMOKE_TEST else 3
N_BOOT = 200 if SMOKE_TEST else 2000
UNET_EPOCHS = 2 if SMOKE_TEST else 20

N_CLIENTS = 6 if SMOKE_TEST else 18
PRIMARY_ALPHA = 0.3
ALPHAS = [0.3, 1.0] if SMOKE_TEST else [0.1, 0.3, 1.0]
FED_SEEDS = 3 if SMOKE_TEST else 20
FED_ROUNDS = 2 if SMOKE_TEST else 60

OBJECTIVE_ROUNDS = 2 if SMOKE_TEST else 40
OBJECTIVE_BATCH = 32
OBJECTIVE_LR = 0.01
OBJECTIVE_LOCAL_EPOCHS = 1
OBJECTIVE_SEEDS = 2 if SMOKE_TEST else 20
CENTRAL_SEEDS = 2 if SMOKE_TEST else 5

ENT_TEMPLATES = 150 if SMOKE_TEST else 1000
ENT_CANDIDATES = 500 if SMOKE_TEST else 4000
ENT_CANDIDATES_HEAD = 500 if SMOKE_TEST else 3000

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TYPE_NAMES = {0: "meningioma", 1: "glioma", 2: "pituitary"}

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
