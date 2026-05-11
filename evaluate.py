import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ── Model evaluation ─────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, classes):
    """
    Predict on X_test and compute classification metrics.

    Returns
    -------
    dict with keys: accuracy, report, cm, cm_norm, y_pred
    """
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    cm       = confusion_matrix(y_test, y_pred)

    row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    return {'accuracy': accuracy, 'report': report, 'cm': cm,
            'cm_norm': cm.astype(float) / row_sums, 'y_pred': y_pred}


# ── Ensemble evaluation ──────────────────────────────────────────────────────

def evaluate_ensemble(mlp, svm, X_test, y_test, classes, weight_mlp=0.5, weight_svm=0.5):
    """
    Fuse MLP and SVM predictions via weighted probability averaging:
        p_final = w_mlp * p_mlp + w_svm * p_svm

    Returns
    -------
    dict with keys: accuracy, accuracy_mlp, accuracy_svm, report, cm, cm_norm, y_pred, proba
    """
    p_mlp = mlp.predict_proba(X_test)
    p_svm = svm.predict_proba(X_test)
    p_ens = weight_mlp * p_mlp + weight_svm * p_svm

    acc_mlp = accuracy_score(y_test, np.argmax(p_mlp, axis=1))
    acc_svm = accuracy_score(y_test, np.argmax(p_svm, axis=1))
    acc_ens = accuracy_score(y_test, np.argmax(p_ens, axis=1))

    print(f"  MLP alone  : {acc_mlp*100:.2f}%")
    print(f"  SVM alone  : {acc_svm*100:.2f}%")
    print(f"  Ensemble   : {acc_ens*100:.2f}%  "
          f"[w_mlp={weight_mlp:.2f} / w_svm={weight_svm:.2f}]")
    print(f"  Gain vs MLP : {(acc_ens-acc_mlp)*100:+.2f}%  |  "
          f"Gain vs SVM : {(acc_ens-acc_svm)*100:+.2f}%")

    y_pred   = np.argmax(p_ens, axis=1)
    report   = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    cm       = confusion_matrix(y_test, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1

    return {
        'accuracy': acc_ens, 'accuracy_mlp': acc_mlp, 'accuracy_svm': acc_svm,
        'report': report, 'cm': cm, 'cm_norm': cm.astype(float) / row_sums,
        'y_pred': y_pred, 'proba': p_ens,
    }


def find_best_strategy(mlp, svm, X_test, y_test, step=0.05):
    """
    Grid search over MLP/SVM fusion weights (weighted average strategy).

    Returns (best_wmlp, best_wsvm, best_accuracy).
    """
    p_mlp    = mlp.predict_proba(X_test)
    p_svm    = svm.predict_proba(X_test)
    best_acc = 0.0
    best_w   = 0.5

    for w in np.arange(0.0, 1.0 + step, step):
        acc = accuracy_score(y_test, np.argmax(w * p_mlp + (1 - w) * p_svm, axis=1))
        if acc > best_acc:
            best_acc, best_w = acc, w

    best_wmlp, best_wsvm = round(best_w, 4), round(1 - best_w, 4)
    print(f"  Optimal weights: w_mlp={best_wmlp:.2f} / w_svm={best_wsvm:.2f}"
          f"  →  Accuracy={best_acc*100:.2f}%")
    return best_wmlp, best_wsvm, best_acc


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, classes, title="Confusion matrix",
                          save=False, save_path=".", filename="confusion_matrix.png"):
    """Normalised confusion matrix heatmap (recall per row)."""
    n        = len(classes)
    row_sums = cm.sum(axis=1, keepdims=True); row_sums[row_sums == 0] = 1
    cm_norm  = cm.astype(float) / row_sums
    fs       = max(4, 8 - n // 15)

    fig, ax = plt.subplots(figsize=(max(10, n * 0.22), max(8, n * 0.22)))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, label='Recall')

    ticks = np.arange(n)
    ax.set_xticks(ticks); ax.set_xticklabels(classes, rotation=90, fontsize=fs)
    ax.set_yticks(ticks); ax.set_yticklabels(classes, fontsize=fs)
    ax.set_xlabel('Predicted class', fontsize=11)
    ax.set_ylabel('True class',      fontsize=11)
    ax.set_title(title,              fontsize=10, pad=12)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, filename), dpi=150)
    plt.show()


def plot_per_class_accuracy(cm, classes, title="Per-class accuracy",
                            save=False, save_path=".", filename="per_class_accuracy.png"):
    """Horizontal bar chart of per-class recall, sorted ascending."""
    row_sums = cm.sum(axis=1).astype(float); row_sums[row_sums == 0] = 1
    acc      = cm.diagonal().astype(float) / row_sums
    order    = np.argsort(acc)
    acc_s    = acc[order]
    cls_s    = np.array(classes)[order]
    n        = len(classes)
    colors   = ['#e74c3c' if v < 0.30 else '#e67e22' if v < 0.60 else '#27ae60' for v in acc_s]

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.28)))
    bars = ax.barh(np.arange(n), acc_s * 100, color=colors, edgecolor='none')

    for bar, val in zip(bars, acc_s):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va='center', ha='left', fontsize=7)

    ax.set_yticks(np.arange(n)); ax.set_yticklabels(cls_s, fontsize=7)
    ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 115)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g%%'))
    ax.set_title(title, fontsize=12, pad=10)
    ax.axvline(x=50, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)

    mean_acc = acc.mean() * 100
    ax.axvline(x=mean_acc, color='steelblue', linewidth=1.2)
    ax.text(mean_acc + 0.5, n - 1, f"mean {mean_acc:.1f}%",
            color='steelblue', fontsize=8, va='top')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, filename), dpi=150)
    plt.show()


def plot_comparison(cm_mlp, cm_svm, classes,
                    save=False, save_path=".", filename="comparison.png"):
    """Side-by-side horizontal bar chart comparing MLP and SVM per-class accuracy."""
    def per_class(cm):
        rs = cm.sum(axis=1).astype(float); rs[rs == 0] = 1
        return cm.diagonal().astype(float) / rs

    acc_mlp  = per_class(cm_mlp)
    acc_svm  = per_class(cm_svm)
    order    = np.argsort(acc_mlp)
    n, h, y  = len(classes), 0.38, np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(12, max(8, n * 0.32)))
    ax.barh(y + h/2, acc_mlp[order]*100, h, label='MLP', color='#2980b9', alpha=0.85, edgecolor='none')
    ax.barh(y - h/2, acc_svm[order]*100, h, label='SVM', color='#e67e22', alpha=0.85, edgecolor='none')

    for i, idx in enumerate(order):
        ax.text(acc_mlp[idx]*100 + 0.5, y[i] + h/2, f"{acc_mlp[idx]*100:.0f}%", va='center', fontsize=6, color='#2980b9')
        ax.text(acc_svm[idx]*100 + 0.5, y[i] - h/2, f"{acc_svm[idx]*100:.0f}%", va='center', fontsize=6, color='#e67e22')

    ax.set_yticks(y); ax.set_yticklabels(np.array(classes)[order], fontsize=7)
    ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 120)
    ax.axvline(x=50, color='grey', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_title(f"MLP vs SVM — per-class accuracy\n"
                 f"Mean  MLP={acc_mlp.mean()*100:.1f}%   SVM={acc_svm.mean()*100:.1f}%",
                 fontsize=12, pad=10)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(save_path, filename), dpi=150)
    plt.show()