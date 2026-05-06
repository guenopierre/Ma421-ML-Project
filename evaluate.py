"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
Évaluation du modèle entraîné et visualisations :
  - Accuracy globale
  - Rapport de classification (précision, rappel, F1 par classe)
  - Matrice de confusion (normalisée par ligne → recall par classe)
  - Histogramme de l'accuracy par classe
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Calcul des métriques
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, classes):
    """
    Prédit sur X_test et calcule toutes les métriques.

    Retourne un dict :
      accuracy  : float
      report    : str  (classification_report)
      cm        : ndarray (n_classes, n_classes) — confusion matrix brute
      cm_norm   : ndarray — confusion matrix normalisée par ligne (recall)
      y_pred    : ndarray — prédictions
    """
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=classes,
                                     zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Normalisation par ligne (recall par classe)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    return {
        'accuracy': accuracy,
        'report':   report,
        'cm':       cm,
        'cm_norm':  cm_norm,
        'y_pred':   y_pred,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Évaluation avec Test-Time Augmentation (TTA)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model_tta(model, X_test_nobg, X_test_bg, y_test, classes):
    """
    Évaluation avec Test-Time Augmentation (TTA).

    Pour chaque image de test, on dispose de deux versions :
      - X_test_nobg : image sans fond (traitée par rembg)
      - X_test_bg   : image originale (avec fond, crop + resize uniquement)

    Le modèle produit des probabilités pour chacune des deux versions.
    On fait la moyenne des deux vecteurs de probabilités, puis on prend
    la classe avec la probabilité moyenne la plus haute.

    Pourquoi ça aide ?
      - Si rembg a mal découpé un avion, la version avec fond corrige le tir
      - Si le fond est trop chargé, la version sans fond est plus fiable
      - La moyenne réduit la variance de prédiction

    Retourne le même dict que evaluate_model, avec en plus :
      accuracy_nobg : accuracy sur images sans fond seules
      accuracy_bg   : accuracy sur images avec fond seules
      accuracy_tta  : accuracy après fusion (= accuracy principale)
    """
    # Probabilités pour chaque version
    proba_nobg = model.predict_proba(X_test_nobg)   # (n_test, n_classes)
    proba_bg   = model.predict_proba(X_test_bg)     # (n_test, n_classes)

    # Fusion : moyenne des probabilités
    proba_tta  = (proba_nobg + proba_bg) / 2.0

    # Prédictions
    y_pred_nobg = np.argmax(proba_nobg, axis=1)
    y_pred_bg   = np.argmax(proba_bg,   axis=1)
    y_pred_tta  = np.argmax(proba_tta,  axis=1)

    acc_nobg = accuracy_score(y_test, y_pred_nobg)
    acc_bg   = accuracy_score(y_test, y_pred_bg)
    acc_tta  = accuracy_score(y_test, y_pred_tta)

    print(f"  Accuracy sans fond seul  : {acc_nobg*100:.2f}%")
    print(f"  Accuracy avec fond seul  : {acc_bg*100:.2f}%")
    print(f"  Accuracy TTA (moyenne)   : {acc_tta*100:.2f}%  ← résultat final")

    report = classification_report(y_test, y_pred_tta,
                                   target_names=classes,
                                   zero_division=0)
    cm = confusion_matrix(y_test, y_pred_tta)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    return {
        'accuracy':      acc_tta,
        'accuracy_nobg': acc_nobg,
        'accuracy_bg':   acc_bg,
        'report':        report,
        'cm':            cm,
        'cm_norm':       cm_norm,
        'y_pred':        y_pred_tta,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Matrice de confusion
# ═══════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, classes,
                          title     = "Matrice de confusion",
                          save      = False,
                          save_path = ".",
                          filename  = "confusion_matrix.png"):
    """
    Affiche la matrice de confusion normalisée (recall par classe).

    La diagonale = accuracy par classe (recall).
    Les cases hors-diagonale montrent les confusions fréquentes.
    """
    n = len(classes)

    # Normalisation par ligne
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(max(10, n * 0.22), max(8, n * 0.22)))

    im = ax.imshow(cm_norm, interpolation='nearest',
                   cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, label='Recall')

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=90, fontsize=max(4, 8 - n // 15))
    ax.set_yticklabels(classes,               fontsize=max(4, 8 - n // 15))
    ax.set_xlabel('Classe prédite',  fontsize=11)
    ax.set_ylabel('Classe réelle',   fontsize=11)
    ax.set_title(title,              fontsize=10, pad=12)

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(save_path, filename), dpi=150)
        print(f"  → Figure sauvegardée : {filename}")

    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Accuracy par classe (histogramme horizontal)
# ═══════════════════════════════════════════════════════════════════════════

def plot_per_class_accuracy(cm, classes,
                            title     = "Accuracy par classe",
                            save      = False,
                            save_path = ".",
                            filename  = "per_class_accuracy.png"):
    """
    Histogramme horizontal de l'accuracy (recall) pour chaque classe.
    Les classes sont triées par accuracy croissante pour repérer facilement
    les classes difficiles.
    """
    # Recall = TP / (TP + FN) = diagonale / somme de la ligne
    row_sums = cm.sum(axis=1).astype(float)
    row_sums[row_sums == 0] = 1
    per_class_acc = cm.diagonal().astype(float) / row_sums

    # Tri par accuracy croissante
    order = np.argsort(per_class_acc)
    sorted_acc     = per_class_acc[order]
    sorted_classes = np.array(classes)[order]

    n = len(classes)
    fig_height = max(6, n * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Couleur : rouge si < 30 %, orange si < 60 %, vert sinon
    colors = ['#e74c3c' if v < 0.30 else
              '#e67e22' if v < 0.60 else
              '#27ae60'
              for v in sorted_acc]

    bars = ax.barh(np.arange(n), sorted_acc * 100, color=colors, edgecolor='none')

    # Valeur en bout de barre
    for i, (bar, val) in enumerate(zip(bars, sorted_acc)):
        ax.text(bar.get_width() + 0.5, i, f"{val*100:.1f}%",
                va='center', ha='left', fontsize=7)

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(sorted_classes, fontsize=7)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 115)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax.set_title(title, fontsize=12, pad=10)
    ax.axvline(x=50, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)

    # Ligne de moyenne globale
    mean_acc = per_class_acc.mean() * 100
    ax.axvline(x=mean_acc, color='steelblue', linestyle='-', linewidth=1.2)
    ax.text(mean_acc + 0.5, n - 1, f"moy. {mean_acc:.1f}%",
            color='steelblue', fontsize=8, va='top')

    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(save_path, filename), dpi=150)
        print(f"  → Figure sauvegardée : {filename}")

    plt.show()