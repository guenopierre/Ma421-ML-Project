import time, importlib
import numpy as np

# ─── Paramètres ──────────────────────────────────────────────────────────
DATA_PATH       = "."
ANNOTATION_FILE = "images_family_trainval.txt"
IMG_SIZE        = 128
CROP_BOTTOM     = 20
USE_REMBG       = True
AUGMENT_WITH_BG = True
BALANCE_CLASSES = False
TARGET_PER_CLASS= None
NUM_PCS         = 300
TEST_SPLIT      = 0.1
RANDOM_STATE    = 42
CACHE_DIR       = "images_withoutback"

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

# ─── Imports avec reload (Spyder) ────────────────────────────────────────
import preprocess, train_mlp, train_svm, evaluate
for mod in [preprocess, train_mlp, train_svm, evaluate]:
    importlib.reload(mod)

from preprocess import run_preprocessing
from train_mlp  import train_mlp
from train_svm  import train_svm
from evaluate   import (evaluate_model, evaluate_ensemble, find_best_strategy,
                        plot_confusion_matrix, plot_per_class_accuracy, plot_comparison)

# ─── Préprocessing ───────────────────────────────────────────────────────
ACTIVE_CLASSES = [c for c, keep in CLASSES_TO_KEEP.items() if keep]
excluded       = [c for c, keep in CLASSES_TO_KEEP.items() if not keep]
print(f"\n{'='*50}")
print(f"  Preprocessing")
print(f"  {len(ACTIVE_CLASSES)} classes | {len(excluded)} exclues : {excluded}")
print(f"{'='*50}")

data    = run_preprocessing(
    data_path=DATA_PATH, annotation_file=ANNOTATION_FILE,
    img_size=IMG_SIZE, crop_bottom=CROP_BOTTOM,
    use_rembg=USE_REMBG, augment_with_bg=AUGMENT_WITH_BG,
    do_balance=BALANCE_CLASSES, target_per_class=TARGET_PER_CLASS,
    num_pcs=NUM_PCS, test_split=TEST_SPLIT,
    random_state=RANDOM_STATE, cache_dir=CACHE_DIR,
    classes_to_keep=ACTIVE_CLASSES,
)

X_train      = data["X_train"]
X_test       = data["X_test"]
X_nobg_train = data["X_nobg_train"]
y_train      = data["y_train"]
y_nobg_train = data["y_nobg_train"]
y_test       = data["y_test"]
classes      = data["classes"]
print(f"  Train={X_train.shape[0]} | Test={X_test.shape[0]} | PCA={X_train.shape[1]}")

# ─── MLP ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("  Reseau de neurones en cours... cela prend environ 4min")
print(f"{'='*50}")
t0  = time.time()
mlp = train_mlp(X_train, y_train, random_state=RANDOM_STATE)
t_mlp = time.time() - t0

# ─── SVM ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("  SVM en cours... 11min environ")
print(f"{'='*50}")
t0  = time.time()
svm = train_svm(X_nobg_train, y_nobg_train, random_state=RANDOM_STATE)
t_svm = time.time() - t0

# ─── Évaluation ──────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print("  Resultats")
print(f"{'='*50}")

results_mlp = evaluate_model(mlp, X_test, y_test, classes)
results_svm = evaluate_model(svm, X_test, y_test, classes)

best_strategy, best_wmlp, best_wsvm, best_acc = find_best_strategy(mlp, svm, X_test, y_test)

results_ens = evaluate_ensemble(mlp, svm, X_test, y_test, classes,
                                weight_mlp=best_wmlp, weight_svm=best_wsvm,
                                strategy=best_strategy)

print(f"\n  Accuracy MLP      : {results_mlp['accuracy']*100:.2f}%  ({t_mlp:.0f}s)")
print(f"  Accuracy SVM      : {results_svm['accuracy']*100:.2f}%  ({t_svm:.0f}s)")
print(f"  Accuracy Ensemble : {results_ens['accuracy']*100:.2f}%  "
      f"[{best_wmlp:.2f}*MLP + {best_wsvm:.2f}*SVM")

# ─── Visualisations ──────────────────────────────────────────────────────
tag = f"IMG={IMG_SIZE}px | PCA={NUM_PCS}"
plot_confusion_matrix(results_mlp["cm"], classes,
    title=f"MLP ({results_mlp['accuracy']*100:.1f}%) — {tag}")
plot_confusion_matrix(results_svm["cm"], classes,
    title=f"SVM ({results_svm['accuracy']*100:.1f}%) — {tag}")
plot_confusion_matrix(results_ens["cm"], classes,
    title=f"Ensemble ({results_ens['accuracy']*100:.1f}%) [{best_strategy}] — {tag}")
plot_comparison(results_mlp["cm"], results_svm["cm"], classes)
plot_per_class_accuracy(results_ens["cm"], classes,
    title=f"Accuracy par classe — Ensemble ({results_ens['accuracy']*100:.1f}%)")


# pour l'interface :
    
import joblib

# À ajouter après l'entraînement et l'évaluation
print("\n Sauvegarde du modèle pour l'interface...")
model_to_save = {
    'data': data,
    'mlp': mlp,
    'svm': svm,
    'CROP_BOTTOM': CROP_BOTTOM,
    'IMG_SIZE': IMG_SIZE
}
joblib.dump(model_to_save, "model_aircraft.pkl")
print("Modèle sauvegardé dans 'model_aircraft.pkl'")