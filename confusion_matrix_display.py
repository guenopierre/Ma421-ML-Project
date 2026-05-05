import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches




def confusion_matrix_display(cm_mlp, y_test, num_classes, classes, title, save=False, DATA_PATH=".", nom_fichier='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_mlp, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=len(y_test) // num_classes)
    plt.colorbar(im)
    
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(tick_marks, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(tick_marks, fontsize=7)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(DATA_PATH, nom_fichier), dpi=150)
        print(f"  → Figure saved: {nom_fichier}")
    plt.show()


def legend_display(classes, save=False, DATA_PATH=".", nom_fichier='legend.png'):
    num_classes = len(classes)
    
    # Hauteur dynamique selon le nombre de classes
    fig_height = max(4, num_classes * 0.28)
    fig, ax = plt.subplots(figsize=(5, fig_height))
    ax.axis('off')
    
    ax.set_title("Légende des classes", fontweight='bold', fontsize=12, pad=10)
    
    lines = ["N°  —  Classe", ""]
    for i, name in enumerate(classes):
        lines.append(f"{i:>3}  —  {name}")
    
    text = "\n".join(lines)
    ax.text(0.05, 0.98, text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace')
    
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(DATA_PATH, nom_fichier), dpi=150)
        print(f"  → Figure saved: {nom_fichier}")
    plt.show()
