# ouais les jeunes 
#ouai la ouai

import numpy as np
import matplotlib.pyplot as plt
import os


donnees = []

with open(r"C:\Users\morga\OneDrive - IPSA\Bureau\Aero4\machine_learning\Ma421-ML-Project\images_family_trainval.txt", "r") as f:
    for line in f:
        partis = line.strip().split(" ",1) # on sépare au preier esapce
        
        donnees.append([partis[0], partis[1]])

donnees = np.array(donnees)

def preprocess_image(image_path, re_size= (500,500)):
    """
    Charge une image, la convertit en niveaux de gris, supprime 20 pixels en bas,
    et la redimensionne en 500x500.
    """
    # Charger l'image en niveaux de gris
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")
    
    # Supprimer 20 pixels en bas
    height, width = img.shape
    if height <= 20:
        raise ValueError("L'image est trop petite pour supprimer 20 pixels en bas.")
    img_cropped = img[:-20, :]
    
    # Redimensionner en 500x500
    img_resized = cv2.resize(img_cropped, re_size)
    
    return img_resized


def ACP_dim2(data, labels_lignes=None, labels_colonnes=None):
    """
    Réalise une ACP en 2 dimensions sur un jeu de données.

    Paramètres :
    - data : tableau numpy de taille (m, n) contenant les données brutes.
    - labels_lignes : liste des labels pour chaque individu (ligne).
    - labels_colonnes : liste des labels pour chaque variable (colonne).

    Retourne :
    - P : matrice des coordonnées des individus projetés sur les 2 premiers axes.
    - C : matrice des contributions des variables aux 2 premiers axes.
    - L : liste des taux d'inertie expliquée par les 2 premiers axes.
    """

    # 1. Centrage et réduction des données
    m, n = data.shape
    moyennes = np.mean(data, axis=0)
    data_centree = data - moyennes
    ecarts_types = np.std(data_centree, axis=0, ddof=0)
    R = data_centree / ecarts_types

    # 2. Décomposition en valeurs singulières (SVD)
    U, S, VT = np.linalg.svd(R, full_matrices=False)

    # 3. Calcul des coordonnées des individus projetés (P)
    P = U[:, :2] @ np.diag(S[:2])

    # 4. Calcul des contributions des variables (C)
    C = (1/(m*n)) * VT.T @ np.diag(S**2)
    C = C[:2, :].T  # On ne garde que les 2 premières composantes

    # 5. Calcul des taux d'inertie expliquée (L)
    L = (S**2) / (m*n)
    L = L[:2]

    # 6. Visualisation
    cmap = plt.get_cmap("tab10")
    if labels_lignes is None:
        labels_lignes = np.arange(m)
    labels_colors = {i: labels_lignes[i] for i in range(m)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle("Analyse en composantes principales en dimension 2", fontsize=16, fontweight="bold")

    # Projection des individus
    ax1.set_title("Représentation des données projetées", fontsize=14, fontweight="bold")
    for i in range(m):
        ax1.scatter(P[i, 0], P[i, 1], c=cmap(labels_colors[i]), label=labels_lignes[i])
    ax1.set_xlabel(f'$v_1$ (taux explicatif {100*L[0]:.2f}%)', fontsize=12)
    ax1.set_ylabel(f'$v_2$ (taux explicatif {100*L[1]:.2f}%)', fontsize=12)
    ax1.grid(True)

    # Cercle des corrélations
    ax2.set_title("Cercle des corrélations", fontsize=14, fontweight="bold")
    for i in range(n):
        ax2.arrow(0, 0, C[i, 0], C[i, 1], color='b', alpha=0.7, head_width=0.05)
        ax2.text(C[i, 0]*1.15, C[i, 1]*1.15, labels_colonnes[i] if labels_colonnes else f'Var {i+1}', color='b', ha='center', va='center')
    circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--')
    ax2.add_patch(circle)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect('equal')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return P, C, L