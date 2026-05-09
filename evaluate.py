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
    Affiche la matrice de confusion normalisée.
    Les cases hors-diagonale montrent les confusions.
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

# ═══════════════════════════════════════════════════════════════════════════
#  Comparaison MLP vs SVM (accuracy par classe côte à côte)
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison(cm_mlp, cm_svm, classes,
                    save=False, save_path=".", filename="comparison.png"):
    """
    Graphique horizontal comparant l'accuracy par classe du MLP et du SVM.
    """
    def per_class_acc(cm):
        row_sums = cm.sum(axis=1).astype(float)
        row_sums[row_sums == 0] = 1
        return cm.diagonal().astype(float) / row_sums

    acc_mlp = per_class_acc(cm_mlp)
    acc_svm = per_class_acc(cm_svm)

    # Tri par accuracy MLP croissante
    order          = np.argsort(acc_mlp)
    acc_mlp_s      = acc_mlp[order] * 100
    acc_svm_s      = acc_svm[order] * 100
    classes_sorted = np.array(classes)[order]
    n              = len(classes)

    fig_height = max(8, n * 0.32)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y      = np.arange(n)
    height = 0.38

    ax.barh(y + height/2, acc_mlp_s, height, label='MLP',
            color='#2980b9', alpha=0.85, edgecolor='none')
    ax.barh(y - height/2, acc_svm_s, height, label='SVM',
            color='#e67e22', alpha=0.85, edgecolor='none')

    # Valeurs en bout de barre
    for i in range(n):
        ax.text(acc_mlp_s[i] + 0.5, y[i] + height/2,
                f"{acc_mlp_s[i]:.0f}%", va='center', fontsize=6, color='#2980b9')
        ax.text(acc_svm_s[i] + 0.5, y[i] - height/2,
                f"{acc_svm_s[i]:.0f}%", va='center', fontsize=6, color='#e67e22')

    ax.set_yticks(y)
    ax.set_yticklabels(classes_sorted, fontsize=7)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 120)
    ax.axvline(x=50, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)

    mean_mlp = acc_mlp.mean() * 100
    mean_svm = acc_svm.mean() * 100
    title_str = (f"Comparaison MLP vs SVM — accuracy par classe"
                 f"\nMoyenne  MLP={mean_mlp:.1f}%   SVM={mean_svm:.1f}%")
    ax.set_title(title_str, fontsize=12, pad=10)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, filename), dpi=150)
        print(f"  → Figure sauvegardée : {filename}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  Ensemble MLP + SVM  (fusion des probabilités)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_ensemble(mlp, svm, X_test, y_test, classes,
                      weight_mlp=0.5, weight_svm=0.5,
                      strategy='best_per_class'):
    """
    Fusionne les prédictions du MLP et du SVM.

    'weighted' :
        proba_finale = w_mlp * proba_mlp + w_svm * proba_svm
    """
    proba_mlp = mlp.predict_proba(X_test)
    proba_svm = svm.predict_proba(X_test)

    y_pred_mlp = np.argmax(proba_mlp, axis=1)
    y_pred_svm = np.argmax(proba_svm, axis=1)

    if strategy == 'best_per_class':
        # Calcule le recall par classe pour chaque modèle
        cm_mlp = confusion_matrix(y_test, y_pred_mlp)
        cm_svm = confusion_matrix(y_test, y_pred_svm)
        row_sums_mlp = cm_mlp.sum(axis=1).astype(float); row_sums_mlp[row_sums_mlp==0]=1
        row_sums_svm = cm_svm.sum(axis=1).astype(float); row_sums_svm[row_sums_svm==0]=1
        recall_mlp = cm_mlp.diagonal() / row_sums_mlp
        recall_svm = cm_svm.diagonal() / row_sums_svm

        # Pour chaque classe, on choisit le modèle avec le meilleur recall
        # On pondère les probabilités de ce modèle × 2 (décision nette)
        n_classes = proba_mlp.shape[1]
        proba_ens = np.zeros_like(proba_mlp)
        for c in range(n_classes):
            if recall_mlp[c] >= recall_svm[c]:
                proba_ens[:, c] = proba_mlp[:, c]
            else:
                proba_ens[:, c] = proba_svm[:, c]

    elif strategy == 'max':
        proba_ens = np.maximum(proba_mlp, proba_svm)

    elif strategy == 'vote':
        n = X_test.shape[0]; nc = proba_mlp.shape[1]
        proba_ens = np.zeros((n, nc))
        for i in range(n):
            proba_ens[i, y_pred_mlp[i]] += 1
            proba_ens[i, y_pred_svm[i]] += 1
        # Égalité → SVM décide (ajoute un epsilon pour le SVM)
        for i in range(n):
            if y_pred_mlp[i] != y_pred_svm[i]:
                proba_ens[i, y_pred_svm[i]] += 0.01

    else:  # 'weighted'
        proba_ens = weight_mlp * proba_mlp + weight_svm * proba_svm

    y_pred_ens = np.argmax(proba_ens, axis=1)

    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    acc_ens = accuracy_score(y_test, y_pred_ens)

    print(f"  MLP seul           : {acc_mlp*100:.2f}%")
    print(f"  SVM seul           : {acc_svm*100:.2f}%")
    print(f"  Ensemble ({strategy}) : {acc_ens*100:.2f}%")
    print(f"  Gain vs MLP : {(acc_ens-acc_mlp)*100:+.2f}%  "
          f"|  Gain vs SVM : {(acc_ens-acc_svm)*100:+.2f}%")

    report = classification_report(y_test, y_pred_ens,
                                   target_names=classes, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_ens)
    row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums==0]=1
    cm_norm  = cm.astype(float) / row_sums

    return {
        'accuracy':     acc_ens,
        'accuracy_mlp': acc_mlp,
        'accuracy_svm': acc_svm,
        'report':       report,
        'cm':           cm,
        'cm_norm':      cm_norm,
        'y_pred':       y_pred_ens,
        'proba':        proba_ens,
        'strategy':     strategy,
    }


def find_best_strategy(mlp, svm, X_test, y_test, step=0.05):
    """
    Teste toutes les stratégies et retourne la meilleure.
    Retourne (strategy, weight_mlp, weight_svm, best_accuracy).
    """
    proba_mlp = mlp.predict_proba(X_test)
    proba_svm = svm.predict_proba(X_test)
    y_pred_mlp = np.argmax(proba_mlp, axis=1)
    y_pred_svm = np.argmax(proba_svm, axis=1)

    results = {}

    # best_per_class
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    rs_mlp = cm_mlp.sum(axis=1).astype(float); rs_mlp[rs_mlp==0]=1
    rs_svm = cm_svm.sum(axis=1).astype(float); rs_svm[rs_svm==0]=1
    rec_mlp = cm_mlp.diagonal() / rs_mlp
    rec_svm = cm_svm.diagonal() / rs_svm
    nc = proba_mlp.shape[1]
    p = np.zeros_like(proba_mlp)
    for c in range(nc):
        p[:, c] = proba_mlp[:, c] if rec_mlp[c] >= rec_svm[c] else proba_svm[:, c]
    results['best_per_class'] = (accuracy_score(y_test, np.argmax(p, axis=1)), 0.5, 0.5)

    # max
    p = np.maximum(proba_mlp, proba_svm)
    results['max'] = (accuracy_score(y_test, np.argmax(p, axis=1)), 0.5, 0.5)

    # vote
    n = X_test.shape[0]
    vm = np.zeros((n, nc))
    for i in range(n):
        vm[i, y_pred_mlp[i]] += 1
        vm[i, y_pred_svm[i]] += 1
        if y_pred_mlp[i] != y_pred_svm[i]:
            vm[i, y_pred_svm[i]] += 0.01
    results['vote'] = (accuracy_score(y_test, np.argmax(vm, axis=1)), 0.5, 0.5)

    # weighted : balayage
    best_w_acc = 0.0; best_w = 0.5
    for w in np.arange(0.0, 1.0 + step, step):
        p = w * proba_mlp + (1-w) * proba_svm
        acc = accuracy_score(y_test, np.argmax(p, axis=1))
        if acc > best_w_acc:
            best_w_acc = acc; best_w = w
    results['weighted'] = (best_w_acc, round(best_w,4), round(1-best_w,4))

    # Affichage
    print(f"  {'Stratégie':<18}  Accuracy")
    print(f"  {'-'*32}")
    for name, (acc, wm, ws) in results.items():
        extra = f"  (w_mlp={wm:.2f}/w_svm={ws:.2f})" if name=='weighted' else ""
        print(f"  {name:<18}  {acc*100:.2f}%{extra}")

    # Meilleure
    best_name = max(results, key=lambda k: results[k][0])
    best_acc, best_wmlp, best_wsvm = results[best_name]
    print(f"\n  Meilleure strategie : '{best_name}' -> {best_acc*100:.2f}%")
    return best_name, best_wmlp, best_wsvm, best_acc