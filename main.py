"""
=============================================================================
 Classification d'avions  —  FGVC-Aircraft (familles)
 Pipeline : chargement → crop → rembg → augmentation → ACP → MLP → TTA
=============================================================================
"""

# ─── Paramètres utilisateur ───────────────────────────────────────────────
DATA_PATH          = "."
ANNOTATION_FILE    = "images_family_trainval.txt"

IMG_SIZE           = 128     # 64=0.6GB  128=2.2GB  256=8.9GB (float32)
CROP_BOTTOM        = 20
USE_REMBG          = True

# Data augmentation train : ajoute les images originales (avec fond)
AUGMENT_WITH_BG    = True

BALANCE_CLASSES    = False
TARGET_PER_CLASS   = None

NUM_PCS            = 150
TEST_SPLIT         = 0.1
RANDOM_STATE       = 42
CACHE_DIR          = "images_withoutback"

# ─── Imports ──────────────────────────────────────────────────────────────
import time
import numpy as np

from preprocess import run_preprocessing
from train      import train_mlp
from evaluate   import (evaluate_model, evaluate_model_tta,
                        plot_confusion_matrix, plot_per_class_accuracy)

# ─── Préprocessing ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ÉTAPE 1 — Préprocessing")
print("="*60)

data = run_preprocessing(
    data_path        = DATA_PATH,
    annotation_file  = ANNOTATION_FILE,
    img_size         = IMG_SIZE,
    crop_bottom      = CROP_BOTTOM,
    use_rembg        = USE_REMBG,
    augment_with_bg  = AUGMENT_WITH_BG,
    do_balance       = BALANCE_CLASSES,
    target_per_class = TARGET_PER_CLASS,
    num_pcs          = NUM_PCS,
    test_split       = TEST_SPLIT,
    random_state     = RANDOM_STATE,
    cache_dir        = CACHE_DIR,
)

X_train    = data["X_train"]
X_test     = data["X_test"]      # test sans fond
X_test_bg  = data["X_test_bg"]   # test avec fond (pour TTA)
y_train    = data["y_train"]
y_test     = data["y_test"]
classes    = data["classes"]

print(f"\n  Train      : {X_train.shape[0]} images")
print(f"  Test nobg  : {X_test.shape[0]} images")
print(f"  Test bg    : {X_test_bg.shape[0]} images")
print(f"  Classes    : {len(classes)}  |  PCA : {X_train.shape[1]} composantes")

# ─── Entraînement MLP ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ÉTAPE 2 — Entraînement du réseau de neurones (MLP)")
print("="*60)

t0 = time.time()
model = train_mlp(X_train, y_train, random_state=RANDOM_STATE)
elapsed = time.time() - t0
print(f"\n  Entraînement terminé en {elapsed:.1f}s")

# ─── Évaluation avec TTA ──────────────────────────────────────────────────
print("\n" + "="*60)
print("  ÉTAPE 3 — Évaluation (Test-Time Augmentation)")
print("="*60)
print("\n  TTA : prédiction sur image sans fond + avec fond → moyenne des probabilités\n")

results = evaluate_model_tta(model, X_test, X_test_bg, y_test, classes)

print(f"\n{results['report']}")

# ─── Visualisations ───────────────────────────────────────────────────────
aug_str = "+TTA" if AUGMENT_WITH_BG else ""
plot_confusion_matrix(
    results["cm"], classes,
    title=(f"Matrice de confusion — MLP+TTA  ({results['accuracy']*100:.1f}%)\n"
           f"nobg={results['accuracy_nobg']*100:.1f}%  "
           f"bg={results['accuracy_bg']*100:.1f}%  "
           f"TTA={results['accuracy']*100:.1f}%  |  PCA={NUM_PCS}")
)
plot_per_class_accuracy(results["cm"], classes,
                        title=f"Accuracy par classe — MLP+TTA ({results['accuracy']*100:.1f}%)")