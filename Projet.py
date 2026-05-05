import time
import matplotlib.pyplot as plt
import importlib, functions_ML_sklearn

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from functions_ML_TD import *
from functions_ML_sklearn import *
from confusion_matrix_display import *



#%% PREPROCESS 


# 0 = N&B; 1 = RGB
images_couleur = 1
 
# 0 = pas d'équilibrage des classes; 1 = sous-échantillonnage pour avoir le même nombre d'images par classe
balance_tri = 1 #pour le train
balance_test = 1 #pour le test
 
# redimensions de nos images (carrées)
IMG_SIZE = 128

# ACP
own_acp = 0 #indiquer 1 si on utilise la pca du cours (prend bcp bcp de temps)
NUM_PCS  = 200 #composantes principales gardées
 
from preprocess import run_preprocessing
 
pp = run_preprocessing(
    images_couleur = images_couleur,
    balance_tri    = balance_tri,
    balance_test   = balance_test,
    IMG_SIZE       = IMG_SIZE,
    NUM_PCS        = NUM_PCS,
)
 
 
X_train_pca = pp['X_train_pca']
y_train     = pp['y_train']
 
X_test_pca  = pp['X_test_pca']
y_test      = pp['y_test']
 
classes     = pp['classes']
n_train     = pp['n_train']
n_test      = pp['n_test']
num_classes = pp['num_classes']
 
IMG_SIZE    = pp['IMG_SIZE']     
NUM_PCS     = pp['NUM_PCS']
COLOR_SIZE  = pp['COLOR_SIZE']

#%% SVM (Training)

start_time = time.time()
svm_sklearn.fit(X_train_pca, y_train)  #file: functions_ML_sklearn
elapsed = time.time() - start_time
print(f"  Training complete in {elapsed:.2f} seconds.")
 
#%% SVM (Test)

print("Start of the SVM test")
start_time = time.time()

y_pred_test_SVM  = svm_sklearn.predict(X_test_pca) #file: functions_ML_sklearn
test_acc_SVM  = accuracy_score(y_test,  y_pred_test_SVM)

elapsed = time.time() - start_time

print(f"  SVM Test complete in {elapsed:.2f} seconds.")
print(f"  Test accuracy     : {test_acc_SVM:.4f}  ({test_acc_SVM*100:.1f}%) ")

#%% SVM (Test) - Confusion Matrix

cm_SVM = confusion_matrix(y_test, y_pred_test_SVM)
confusion_matrix_display(cm_SVM, y_test, num_classes, classes, f"Matrice de confusion (normalisée) / SVM / {NUM_PCS} composantes principales / images de taille : {COLOR_SIZE} x {IMG_SIZE} x {IMG_SIZE} ", save = False)  #file: confusion_matrix_display

#%% Neural Network - MLP (Training)

importlib.reload(functions_ML_sklearn)

print(f"Hidden Layers: ({hidden_layers})\nMax iterations: {max_iter}")

start_time = time.time()
print("MLP running...")
mlp_sklearn.fit(X_train_pca, y_train) #file: functions_ML_sklearn
elapsed = time.time() - start_time
print(f"  MLP training complete in {elapsed:.2f} seconds.")

#%% Neural Network - MLP (Test)

start_time = time.time()

y_pred_test_mlp  = mlp_sklearn.predict(X_test_pca) #file: functions_ML_sklearn
test_acc_mlp  = accuracy_score(y_test,  y_pred_test_mlp)

elapsed = time.time() - start_time

print(f"  MLP Test complete in {elapsed:.2f} seconds.")
print(f"  Test accuracy     : {test_acc_mlp:.4f}  ({test_acc_mlp*100:.1f}%)")

#%% Neural Network - MLP (Test) - Details

print("\nClassification Report (MLP — Test Set):")
print(classification_report(y_test, y_pred_test_mlp, target_names=classes))

#%% Neural Network - MLP (Test) - Confusion Matrix

cm_mlp = confusion_matrix(y_test, y_pred_test_mlp)
confusion_matrix_display(cm_mlp, y_test, num_classes, classes, f"Matrice de confusion (normalisée) / MLP / accuracy: {test_acc_mlp*100:.1f}% \n hidden layers: {hidden_layers} / {NUM_PCS} composantes principales / images de taille : {COLOR_SIZE} x {IMG_SIZE} x {IMG_SIZE} ", save = False)  #file: confusion_matrix_display
legend_display(classes)

#%% À RUN LA NUIT

import time
import os
import csv
import psutil
import matplotlib.pyplot as plt
import importlib, functions_ML_sklearn

# ─────────────────────────────────────────────
#  PARAMÈTRES FIXES
# ─────────────────────────────────────────────
images_couleur = 1
balance_tri    = 1
balance_test   = 1

# ─────────────────────────────────────────────
#  GRILLE DE RECHERCHE
# ─────────────────────────────────────────────
IMG_SIZE_LIST = [64, 96, 128, 192, 256]
NUM_PCS_LIST  = [20, 50, 75, 100, 120, 150, 200, 300]

# ─────────────────────────────────────────────
#  LIMITATION CPU  (0.0 → 1.0 = fraction max)
# ─────────────────────────────────────────────
CPU_FRACTION  = 0.1   # utilise au max ~25 % du CPU
SLEEP_BETWEEN = 30     # secondes de pause entre chaque run (laisse le CPU refroidir)

def throttle_cpu(fraction: float = 0.25) -> None:
    """
    Abaisse la priorité du processus courant et limite l'affinité
    aux premiers cœurs proportionnellement à `fraction`.
    """
    proc = psutil.Process(os.getpid())

    # ── Priorité basse (syntaxe différente selon l'OS) ──────────────────
    try:
        if os.name == 'nt':  # Windows
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("  [CPU] Priorité abaissée (BELOW_NORMAL) — Windows.")
        else:                # Linux / macOS
            proc.nice(19)
            print("  [CPU] Priorité abaissée (nice=19) — Linux/macOS.")
    except Exception as e:
        print(f"  [CPU] ⚠ Impossible de changer la priorité : {e}")

    # ── Affinité : on ne garde qu'une fraction des cœurs logiques ───────
    total_cores  = psutil.cpu_count(logical=True)
    cores_to_use = max(1, int(total_cores * fraction))
    allowed_cores = list(range(cores_to_use))
    try:
        proc.cpu_affinity(allowed_cores)
        print(f"  [CPU] {cores_to_use}/{total_cores} cœur(s) autorisé(s).")
    except Exception as e:
        print(f"  [CPU] ⚠ Affinité non applicable : {e}")


# ─────────────────────────────────────────────
#  FICHIER DE RÉSULTATS
# ─────────────────────────────────────────────
RESULTS_FILE = "grid_search_results.csv"
fieldnames   = [
    "IMG_SIZE", "NUM_PCS",
    "acc_SVM",  "time_train_SVM",  "time_test_SVM",
    "acc_MLP",  "time_train_MLP",  "time_test_MLP",
]

# Crée le fichier CSV avec l'en-tête si il n'existe pas encore
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# ─────────────────────────────────────────────
#  LIMITATION CPU (appliquée une seule fois)
# ─────────────────────────────────────────────
throttle_cpu(CPU_FRACTION)

# ─────────────────────────────────────────────
#  BOUCLE PRINCIPALE
# ─────────────────────────────────────────────
total_runs    = len(IMG_SIZE_LIST) * len(NUM_PCS_LIST)
current_run   = 0
best_svm      = {"acc": -1, "IMG_SIZE": None, "NUM_PCS": None}
best_mlp      = {"acc": -1, "IMG_SIZE": None, "NUM_PCS": None}

print(f"\n{'='*60}")
print(f"  GRID SEARCH  –  {total_runs} combinaisons à tester")
print(f"{'='*60}\n")

for IMG_SIZE in IMG_SIZE_LIST:
    for NUM_PCS in NUM_PCS_LIST:

        current_run += 1
        print(f"\n[{current_run}/{total_runs}]  IMG_SIZE={IMG_SIZE}  |  NUM_PCS={NUM_PCS}")
        print(f"{'-'*50}")

        # ── Vérifie que NUM_PCS est cohérent avec la taille de l'image ──
        max_possible_pcs = IMG_SIZE * IMG_SIZE * (3 if images_couleur else 1)
        if NUM_PCS > max_possible_pcs:
            print(f"  ⚠  NUM_PCS={NUM_PCS} > dimension max ({max_possible_pcs}) → ignoré.")
            continue

        # ── Préprocessing ────────────────────────────────────────────────
        print("  Preprocessing...")
        from preprocess import run_preprocessing
        pp = run_preprocessing(
            images_couleur = images_couleur,
            balance_tri    = balance_tri,
            balance_test   = balance_test,
            IMG_SIZE       = IMG_SIZE,
            NUM_PCS        = NUM_PCS,
        )

        X_train_pca = pp['X_train_pca']
        y_train     = pp['y_train']
        X_test_pca  = pp['X_test_pca']
        y_test      = pp['y_test']
        classes     = pp['classes']
        num_classes = pp['num_classes']
        COLOR_SIZE  = pp['COLOR_SIZE']

        row = {
            "IMG_SIZE": IMG_SIZE,
            "NUM_PCS":  NUM_PCS,
        }

        # ════════════════════════════════════════
        #  SVM
        # ════════════════════════════════════════
        importlib.reload(functions_ML_sklearn)

        print("  [SVM] Entraînement...")
        t0 = time.time()
        svm_sklearn.fit(X_train_pca, y_train)
        row["time_train_SVM"] = round(time.time() - t0, 2)

        print("  [SVM] Test...")
        t0 = time.time()
        y_pred_svm          = svm_sklearn.predict(X_test_pca)
        row["time_test_SVM"] = round(time.time() - t0, 2)

        row["acc_SVM"] = round(accuracy_score(y_test, y_pred_svm), 4)
        print(f"  [SVM] Accuracy : {row['acc_SVM']*100:.1f}%  "
              f"(train {row['time_train_SVM']}s / test {row['time_test_SVM']}s)")

        if row["acc_SVM"] > best_svm["acc"]:
            best_svm = {"acc": row["acc_SVM"], "IMG_SIZE": IMG_SIZE, "NUM_PCS": NUM_PCS}

        # ════════════════════════════════════════
        #  MLP
        # ════════════════════════════════════════
        print("  [MLP] Entraînement...")
        t0 = time.time()
        mlp_sklearn.fit(X_train_pca, y_train)
        row["time_train_MLP"] = round(time.time() - t0, 2)

        print("  [MLP] Test...")
        t0 = time.time()
        y_pred_mlp           = mlp_sklearn.predict(X_test_pca)
        row["time_test_MLP"] = round(time.time() - t0, 2)

        row["acc_MLP"] = round(accuracy_score(y_test, y_pred_mlp), 4)
        print(f"  [MLP] Accuracy : {row['acc_MLP']*100:.1f}%  "
              f"(train {row['time_train_MLP']}s / test {row['time_test_MLP']}s)")

        if row["acc_MLP"] > best_mlp["acc"]:
            best_mlp = {"acc": row["acc_MLP"], "IMG_SIZE": IMG_SIZE, "NUM_PCS": NUM_PCS}

        # ── Sauvegarde immédiate dans le CSV ─────────────────────────────
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        print(f"  ✔  Résultats sauvegardés dans '{RESULTS_FILE}'")

        # ── Pause pour laisser le CPU souffler ───────────────────────────
        if current_run < total_runs:
            print(f"  💤  Pause de {SLEEP_BETWEEN}s avant le prochain run…")
            time.sleep(SLEEP_BETWEEN)

# ─────────────────────────────────────────────
#  RÉSUMÉ FINAL
# ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("  GRID SEARCH TERMINÉ")
print(f"{'='*60}")
print(f"  Meilleur SVM : {best_svm['acc']*100:.1f}%  "
      f"→  IMG_SIZE={best_svm['IMG_SIZE']}, NUM_PCS={best_svm['NUM_PCS']}")
print(f"  Meilleur MLP : {best_mlp['acc']*100:.1f}%  "
      f"→  IMG_SIZE={best_mlp['IMG_SIZE']}, NUM_PCS={best_mlp['NUM_PCS']}")
print(f"\n  Résultats complets : '{RESULTS_FILE}'\n")

# ─────────────────────────────────────────────
#  VISUALISATION RAPIDE DES RÉSULTATS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np

df = pd.read_csv(RESULTS_FILE)

for model, col in [("SVM", "acc_SVM"), ("MLP", "acc_MLP")]:
    pivot = df.pivot(index="NUM_PCS", columns="IMG_SIZE", values=col)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                   vmin=df[col].min(), vmax=df[col].max())
    plt.colorbar(im, ax=ax, label="Accuracy")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("IMG_SIZE")
    ax.set_ylabel("NUM_PCS")
    ax.set_title(f"Grid Search – Accuracy {model}")

    # Annotations dans chaque cellule
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%",
                        ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"grid_search_{model}.png", dpi=150)
    plt.show()
    
#%%visualisation des résultats de la nuit
    
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Lecture ──────────────────────────────────────────────────
df = pd.read_csv("grid_search_results.csv",
                 header=None,
                 names=["IMG_SIZE", "NUM_PCS",
                        "time_train_SVM", "acc_train_SVM", "acc_test_SVM",
                        "time_train_MLP", "time_test_MLP", "acc_test_MLP"])

# ── Données ──────────────────────────────────────────────────
img_sizes = sorted(df["IMG_SIZE"].unique())
num_pcs   = sorted(df["NUM_PCS"].unique())

x_idx = np.arange(len(img_sizes))
y_idx = np.arange(len(num_pcs))

# ── Figure 3D interactive ────────────────────────────────────
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection="3d")

dx = dy = 0.6
for xi, img in enumerate(img_sizes):
    for yi, npc in enumerate(num_pcs):
        val = df[(df["IMG_SIZE"] == img) & (df["NUM_PCS"] == npc)]["acc_test_MLP"].values
        if len(val) == 0:
            continue
        z = float(val[0])
        color = plt.cm.RdYlGn(z)  # rouge→vert selon accuracy
        ax.bar3d(xi - dx/2, yi - dy/2, 0, dx, dy, z,
                 color=color, edgecolor="grey", linewidth=0.3)

# ── Axes ─────────────────────────────────────────────────────
ax.set_xticks(x_idx)
ax.set_xticklabels(img_sizes)
ax.set_yticks(y_idx)
ax.set_yticklabels(num_pcs)
ax.set_xlabel("IMG_SIZE")
ax.set_ylabel("NUM_PCS")
ax.set_zlabel("Accuracy test MLP")
ax.set_title("MLP — Accuracy test en fonction de IMG_SIZE et NUM_PCS")

plt.tight_layout()
plt.show()  # ← fenêtre interactive, clic+glisser pour tourner



