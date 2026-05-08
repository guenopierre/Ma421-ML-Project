"""
=============================================================================
 Classification d'avions  —  FGVC-Aircraft (familles)
 Pipeline : chargement → crop → rembg → augmentation → ACP → MLP + SVM
=============================================================================
"""

# ─── Paramètres utilisateur ───────────────────────────────────────────────
DATA_PATH          = "."
ANNOTATION_FILE    = "images_family_trainval.txt"

IMG_SIZE           = 128    # 64=0.6GB  128=2.2GB  256=8.9GB (float32)
CROP_BOTTOM        = 20
USE_REMBG          = True
AUGMENT_WITH_BG    = True   # double le train avec images originales

BALANCE_CLASSES    = False
TARGET_PER_CLASS   = None

NUM_PCS            = 500    # composantes PCA 
TEST_SPLIT         = 0.2  
RANDOM_STATE       = 42
CACHE_DIR          = "images_withoutback"


# ─── Sélection des classes (True = inclus, False = exclu) ─────────────────
# Scores indicatifs issus du dernier run (MLP, 128px, rembg)
# Retire une classe en mettant False → le modèle n'apprend pas sur elle
CLASSES_TO_KEEP = {
    "A300":               False,   
    "A310":               True,   
    "A320":               True,   
    "A330":               True,   
    "A340":               True,   
    "A380":               True,   
    "ATR-42":             True,   
    "ATR-72":             True,   
    "An-12":              True,   
    "BAE 146":            True,   
    "BAE-125":            True,
    "Beechcraft 1900":    True,   
    "Boeing 707":         True,   
    "Boeing 717":         True,   
    "Boeing 727":         False,   
    "Boeing 737":         True,   
    "Boeing 747":         True,   
    "Boeing 757":         True,   
    "Boeing 767":         True,   
    "Boeing 777":         True,   
    "C-130":              True,
    "C-47":               True,   
    "CRJ-200":            True,   
    "CRJ-700":            True,   
    "Cessna 172":         True,   
    "Cessna 208":         True,   
    "Cessna Citation":    True,   
    "Challenger 600":     True,   
    "DC-10":              False,
    "DC-3":               True,
    "DC-6":               False,   
    "DC-8":               False,
    "DC-9":               False,   
    "DH-82":              True,   
    "DHC-1":              True,   
    "DHC-6":              False,   
    "DR-400":             True,   
    "Dash 8":             True,   
    "Dornier 328":        True,   
    "EMB-120":            True,   
    "Embraer E-Jet":      True,   
    "Embraer ERJ 145":    True,   
    "Embraer Legacy 600": True,   
    "Eurofighter Typhoon":True,   
    "F-16":               True,   
    "F/A-18":             True,   
    "Falcon 2000":        True,   
    "Falcon 900":         True,   
    "Fokker 100":         True,   
    "Fokker 50":          True,   
    "Fokker 70":          True,   
    "Global Express":     True,   
    "Gulfstream":         True,   
    "Hawk T1":            True,   
    "Il-76":              True,   
    "King Air":           True,   
    "L-1011":             False,   
    "MD-11":              True,   
    "MD-80":              True,   
    "MD-90":              False,   
    "Metroliner":         True,   
    "PA-28":              True,   
    "SR-20":              True,   
    "Saab 2000":          True,   
    "Saab 340":           True,   
    "Spitfire":           True,   
    "Tornado":            True,   
    "Tu-134":             True,   
    "Tu-154":             True,   
    "Yak-42":             True,   
}

# ─── Imports ──────────────────────────────────────────────────────────────
import time
import importlib
import numpy as np

# Force le rechargement des modules à chaque run 
import preprocess, train_mlp, train_svm, evaluate
importlib.reload(preprocess)
importlib.reload(train_mlp)
importlib.reload(train_svm)
importlib.reload(evaluate)

from preprocess  import run_preprocessing
from train_mlp   import train_mlp
from train_svm   import train_svm
from evaluate    import (evaluate_model,
                         evaluate_ensemble,
                         find_best_strategy,
                         plot_confusion_matrix,
                         plot_per_class_accuracy,
                         plot_comparison)

# ─── Préprocessing ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ÉTAPE 1 — Préprocessing")
print("="*60)

# Construit la liste des classes actives à partir du dict True/False
ACTIVE_CLASSES = [c for c, keep in CLASSES_TO_KEEP.items() if keep]
excluded = [c for c, keep in CLASSES_TO_KEEP.items() if not keep]
print(f"  Classes actives   : {len(ACTIVE_CLASSES)} / {len(CLASSES_TO_KEEP)}")
if excluded:
    print(f"  Classes exclues   : {excluded}")

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
    classes_to_keep  = ACTIVE_CLASSES,
)

X_train  = data["X_train"]
X_test   = data["X_test"]
y_train  = data["y_train"]
y_test   = data["y_test"]
classes  = data["classes"]

print(f"\n  Train   : {X_train.shape[0]} images")
print(f"  Test    : {X_test.shape[0]} images")
print(f"  Classes : {len(classes)}  |  PCA : {X_train.shape[1]} composantes")

# ─── Entraînement MLP ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ÉTAPE 2a — MLP (réseau de neurones)")
print("="*60)

t0 = time.time()
mlp = train_mlp(X_train, y_train, random_state=RANDOM_STATE)
t_mlp = time.time() - t0
print(f"\n  MLP entraîné en {t_mlp:.1f}s")

# ─── Entraînement SVM ─────────────────────────────────────────────────────
# Note : le SVM s'entraîne sur les images SANS FOND uniquement (X_nobg)
# car l'augmentation avec fond crée des doublons quasi-identiques qui
# alourdissent la SVM
print("\n" + "="*60)
print("  ÉTAPE 2b — SVM (Support Vector Machine)")
print("="*60)
print("\n  Note : le SVM est entraîné sur les images sans fond uniquement")
print(f"  ({X_train.shape[0]//2} images) — l'augmentation ne lui bénéficie pas.")

X_nobg_train  = data["X_nobg_train"]   # train sans fond (avant augmentation)
y_nobg_train  = data["y_nobg_train"]   # labels correspondants

t0 = time.time()
svm = train_svm(X_nobg_train, y_nobg_train, random_state=RANDOM_STATE)
t_svm = time.time() - t0
print(f"\n  SVM entraîné en {t_svm:.1f}s")

# ─── Évaluation ───────────────────────────────────────────────────────────
# Poids de l'ensemble : ajuste selon les performances observées.
# Ex: si MLP >> SVM, mets WEIGHT_MLP=0.7 et WEIGHT_SVM=0.3
WEIGHT_MLP = 0.5   # poids initial — sera optimisé automatiquement
WEIGHT_SVM = 0.5

# ─── Évaluation individuelle ──────────────────────────────────────────────
print("\n" + "="*60)
print("  ÉTAPE 3 — Évaluation individuelle MLP et SVM")
print("="*60)

results_mlp = evaluate_model(mlp, X_test, y_test, classes)
results_svm = evaluate_model(svm, X_test, y_test, classes)

print(f"\n  {'Modèle':<8}  {'Accuracy':>10}  {'Temps train':>12}")
print(f"  {'-'*35}")
print(f"  {'MLP':<8}  {results_mlp['accuracy']*100:>9.2f}%  {t_mlp:>10.1f}s")
print(f"  {'SVM':<8}  {results_svm['accuracy']*100:>9.2f}%  {t_svm:>10.1f}s")


print("  weighted : moyenne pondérée (tous poids 0.0→1.0 par 0.05)\n")

best_strategy, best_wmlp, best_wsvm, best_acc = find_best_strategy(
    mlp, svm, X_test, y_test
)

# ─── Évaluation ensemble avec la meilleure stratégie ─────────────────────
print("\n" + "="*60)
print(f"  ÉTAPE 5 — Ensemble MLP + SVM  (stratégie : {best_strategy})")
print("="*60)

results_ens = evaluate_ensemble(mlp, svm, X_test, y_test, classes,
                                weight_mlp=best_wmlp,
                                weight_svm=best_wsvm,
                                strategy=best_strategy)


print(f"\n── Rapport Ensemble ──\n{results_ens['report']}")

# ─── Visualisations ───────────────────────────────────────────────────────
tag = f"IMG={IMG_SIZE}px | rembg | PCA={NUM_PCS}"

plot_confusion_matrix(
    results_mlp["cm"], classes,
    title=f"Matrice de confusion — MLP ({results_mlp['accuracy']*100:.1f}%)\n{tag}"
)
plot_confusion_matrix(
    results_svm["cm"], classes,
    title=f"Matrice de confusion — SVM ({results_svm['accuracy']*100:.1f}%)\n{tag}"
)
plot_confusion_matrix(
    results_ens["cm"], classes,
    title=(f"Matrice de confusion — Ensemble MLP+SVM ({results_ens['accuracy']*100:.1f}%) [{best_strategy}]\n"
           f"MLP={results_ens['accuracy_mlp']*100:.1f}%  "
           f"SVM={results_ens['accuracy_svm']*100:.1f}%  "
           f"w={best_wmlp:.2f}/{best_wsvm:.2f}")
)
plot_comparison(results_mlp["cm"], results_svm["cm"], classes)
plot_per_class_accuracy(
    results_ens["cm"], classes,
    title=f"Accuracy par classe — Ensemble MLP+SVM ({results_ens['accuracy']*100:.1f}%)"
)