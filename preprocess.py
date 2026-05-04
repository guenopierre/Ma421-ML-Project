#%% LIBRAIRIES

import os
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from functions_preprocess import (
    load_annotations,
    balance_classes,
    load_images_gray,
    load_images_color,
)


#%% Fonction principale appelée depuis le main
# -----------------------------------------------------------------------------
# Cette fonction remplace les 4 anciens fichiers preprocess_*.py.
# Le comportement est piloté par les flags `images_couleur`, `balance_tri` et
# `balance_test`, et les hyper-paramètres `IMG_SIZE` et `NUM_PCS` peuvent être
# passés directement depuis le main.
# -----------------------------------------------------------------------------

def run_preprocessing(images_couleur=1,
                      balance_tri=1,
                      balance_test=0,          # <-- nouveau flag
                      IMG_SIZE=None,
                      NUM_PCS=100,
                      target_per_class=None,
                      target_per_class_test=None,   # <-- nouveau paramètre
                      data_path='.',
                      random_state=42):
    """
    Pipeline complet de préprocessing :
      1. Lecture des annotations train/test
      2. (optionnel) Équilibrage des classes du train
      3. (optionnel) Équilibrage des classes du test
      4. Encodage des labels
      5. Chargement + flatten + normalisation des images (gray ou color)
      6. Centrage avec la moyenne du train
      7. PCA fit sur le train, appliquée au test

    Parameters
    ----------
    images_couleur : int
        0 = images en niveaux de gris  (avec crop du bandeau de crédit)
        1 = images en couleur RGB      (façon Matlab, sans crop)
    balance_tri : int
        0 = pas d'équilibrage des classes du train
        1 = équilibrage du train par sous-échantillonnage
    balance_test : int
        0 = pas d'équilibrage des classes du test  (comportement par défaut,
            le test reflète la distribution réelle)
        1 = équilibrage du test par sous-échantillonnage
    IMG_SIZE : int or None
        Taille du carré (IMG_SIZE × IMG_SIZE) auquel sont redimensionnées les
        images. Si None, vaut 128 en N&B et 64 en couleur.
    NUM_PCS : int
        Nombre de composantes principales à conserver après PCA.
    target_per_class : int or None
        Si balance_tri=1, nombre d'images cible par classe dans le train.
        None = taille de la classe la plus petite du train.
    target_per_class_test : int or None
        Si balance_test=1, nombre d'images cible par classe dans le test.
        None = taille de la classe la plus petite du test.
    data_path : str
        Chemin vers le dossier contenant les fichiers d'annotation et 'images/'.
    random_state : int
        Graine aléatoire pour l'équilibrage.

    Returns
    -------
    out : dict
        Dictionnaire contenant toutes les sorties utiles au main :
        X_train_pca, y_train, X_test_pca, y_test,
        classes, n_train, n_test, num_classes,
        IMG_SIZE, NUM_PCS, COLOR_SIZE
    """

    #%% STEP 1 — Configuration

    TRAIN_ANNOT = os.path.join(data_path, 'images_family_train.txt')
    TEST_ANNOT  = os.path.join(data_path, 'images_family_test.txt')
    IMAGE_DIR   = os.path.join(data_path, 'images')

    # COLOR_SIZE : 1 canal (N&B) ou 3 canaux (RGB)
    if images_couleur == 1:
        COLOR_SIZE = 3
        if IMG_SIZE is None:
            IMG_SIZE = 64
    else:
        COLOR_SIZE = 1
        if IMG_SIZE is None:
            IMG_SIZE = 128

    print(f"\n=== Preprocessing ===")
    print(f"  images_couleur = {images_couleur}  (COLOR_SIZE={COLOR_SIZE})")
    print(f"  balance_tri    = {balance_tri}")
    print(f"  balance_test   = {balance_test}")
    print(f"  IMG_SIZE       = {IMG_SIZE}")
    print(f"  NUM_PCS        = {NUM_PCS}")

    #%% STEP 2 — Load Annotations

    train_imgs, labels_train = load_annotations(TRAIN_ANNOT)
    test_imgs,  labels_test  = load_annotations(TEST_ANNOT)

    # --- Équilibrage du train ---
    if balance_tri == 1:
        print("\nBalancing training classes...")
        train_imgs, labels_train = balance_classes(
            train_imgs, labels_train,
            target_count=target_per_class,
            random_state=random_state
        )

    # --- Équilibrage du test (optionnel) ---
    # Attention : équilibrer le test modifie la distribution réelle des classes
    # et peut biaiser les métriques d'évaluation. À utiliser avec précaution,
    # par exemple pour comparer les classes à iso-effectif.
    if balance_test == 1:
        print("\nBalancing test classes...")
        test_imgs, labels_test = balance_classes(
            test_imgs, labels_test,
            target_count=target_per_class_test,
            random_state=random_state
        )

    # Encodage des labels (ex: 'A330' -> 3), ordre alphabétique
    le = LabelEncoder()
    le.fit(labels_train + labels_test)
    y_train = le.transform(labels_train)
    y_test  = le.transform(labels_test)

    classes     = le.classes_
    n_train     = len(train_imgs)
    n_test      = len(test_imgs)
    num_classes = len(classes)

    #%% STEP 3 — Feature Extraction (resize + flatten + normalize)
    #
    # Branche GRAY :
    #   1. Load image in grayscale (cv2.IMREAD_GRAYSCALE)
    #   2. Remove the bottom 20 pixels (photo credit banner)
    #   3. Resize to IMG_SIZE x IMG_SIZE
    #   4. Flatten to a 1D vector and normalize to [0, 1]
    #   --> chaque image produit un vecteur de taille IMG_SIZE*IMG_SIZE
    #
    # Branche COLOR (façon Matlab) :
    #   1. Load image in COLOR (3 channels, RGB)
    #   2. Resize directly to IMG_SIZE x IMG_SIZE  (pas de crop)
    #   3. Flatten in Fortran order (column-major, comme Matlab) et normalize to [0, 1]
    #   --> chaque image produit un vecteur de taille IMG_SIZE*IMG_SIZE*3

    print("\nLoading training images...")
    if images_couleur == 1:
        X_train = load_images_color(train_imgs, IMAGE_DIR, IMG_SIZE)
    else:
        X_train = load_images_gray(train_imgs, IMAGE_DIR, IMG_SIZE)

    print("Loading test images...")
    if images_couleur == 1:
        X_test = load_images_color(test_imgs, IMAGE_DIR, IMG_SIZE)
    else:
        X_test = load_images_gray(test_imgs, IMAGE_DIR, IMG_SIZE)

    print(f"  X_train shape: {X_train.shape}  (n_samples × n_features)")
    print(f"  X_test  shape: {X_test.shape}")

    #%% STEP 4 — Centrage avec la moyenne du TRAIN
    #
    #   mu = mean(X_train)          <- calculée sur le train uniquement
    #   X_train_c = X_train - mu
    #   X_test_c  = X_test  - mu    <- on soustrait la MÊME moyenne du train
    #
    # Le test n'intervient jamais dans le calcul de mu : cela évite toute
    # fuite d'information (data leakage) du test vers le train.

    mu = X_train.mean(axis=0)
    X_train_c = X_train - mu
    X_test_c  = X_test  - mu

    #%% STEP 5 — PCA for Dimensionality Reduction

    pca = PCA(n_components=NUM_PCS)
    X_train_pca = pca.fit_transform(X_train_c)   # fit + projection sur le train
    X_test_pca  = pca.transform(X_test_c)        # projection sur les c.p. du train

    #%% Sortie

    return {
        'X_train_pca': X_train_pca,
        'y_train':     y_train,
        'X_test_pca':  X_test_pca,
        'y_test':      y_test,
        'classes':     classes,
        'n_train':     n_train,
        'n_test':      n_test,
        'num_classes': num_classes,
        'IMG_SIZE':    IMG_SIZE,
        'NUM_PCS':     NUM_PCS,
        'COLOR_SIZE':  COLOR_SIZE,
        'pca':         pca,
        'mu':          mu,
    }
