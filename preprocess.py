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
# Le comportement est piloté par les flags `images_couleur` et `balance_tri`,
# et les hyper-paramètres `IMG_SIZE` et `NUM_PCS` peuvent être passés
# directement depuis le main (c'est ce qui ne marchait pas avant : quand on
# fait `import preprocess`, le code du fichier s'exécute UNE SEULE FOIS au
# moment de l'import, donc IMG_SIZE / NUM_PCS étaient figés. En passant par
# une fonction, on peut au contraire les changer à chaque appel).
# -----------------------------------------------------------------------------

def run_preprocessing(images_couleur=1,
                      balance_tri=1,
                      IMG_SIZE=None,
                      NUM_PCS=100,
                      target_per_class=None,
                      data_path='.',
                      random_state=42):
    """
    Pipeline complet de préprocessing :
      1. Lecture des annotations train/test
      2. (optionnel) Équilibrage des classes du train
      3. Encodage des labels
      4. Chargement + flatten + normalisation des images (gray ou color)
      5. (color only, façon Matlab) Centrage avec la moyenne du train
      6. PCA fit sur le train, appliquée au test

    Parameters
    ----------
    images_couleur : int
        0 = images en niveaux de gris  (avec crop du bandeau de crédit)
        1 = images en couleur RGB      (façon Matlab, sans crop)
    balance_tri : int
        0 = pas d'équilibrage des classes
        1 = équilibrage par sous-échantillonnage
    IMG_SIZE : int or None
        Taille du carré (IMG_SIZE × IMG_SIZE) auquel sont redimensionnées les
        images. Si None, vaut 128 en N&B et 64 en couleur (= valeurs par défaut
        des anciens fichiers).
    NUM_PCS : int
        Nombre de composantes principales à conserver après PCA.
    target_per_class : int or None
        Si balance_tri=1, nombre d'images cible par classe.
        None = taille de la classe la plus petite.
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
        # IMG_SIZE par défaut = 64 pour la version couleur (comme Matlab)
        if IMG_SIZE is None:
            IMG_SIZE = 64
    else:
        COLOR_SIZE = 1
        # IMG_SIZE par défaut = 128 pour la version N&B (comme l'ancien preprocess.py)
        if IMG_SIZE is None:
            IMG_SIZE = 128

    print(f"\n=== Preprocessing ===")
    print(f"  images_couleur = {images_couleur}  (COLOR_SIZE={COLOR_SIZE})")
    print(f"  balance_tri    = {balance_tri}")
    print(f"  IMG_SIZE       = {IMG_SIZE}")
    print(f"  NUM_PCS        = {NUM_PCS}")

    #%% STEP 2 — Load Annotations

    train_imgs, labels_train = load_annotations(TRAIN_ANNOT)
    test_imgs,  labels_test  = load_annotations(TEST_ANNOT)

    # /!\ On n'équilibre PAS le test : il doit refléter la distribution réelle
    if balance_tri == 1:
        print("\nBalancing training classes...")
        train_imgs, labels_train = balance_classes(
            train_imgs, labels_train,
            target_count=target_per_class,
            random_state=random_state
        )

    # convertir le label (ex: 'A330') en entier (ex: 3)
    # --> il le fait par ordre alphabétique
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
    #   2. Remove the bottom 20 pixels (photo credit banner, useless for the algo)
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

    #%% STEP 4 — Centrage avec la moyenne du TRAIN (uniquement pour la branche couleur, façon Matlab)
    #
    # Matlab fait :
    #   mu = mean(X_train);
    #   X_train_c = X_train - mu;
    #   X_test_c  = X_test  - mu;
    # On reproduit exactement ça, puis PCA sur les données CENTRÉES.
    #
    # En N&B on ne le faisait pas dans l'ancien preprocess.py (sklearn PCA
    # centre déjà en interne), donc on garde le comportement d'origine.

    if images_couleur == 1:
        mu = X_train.mean(axis=0)
        X_train_c = X_train - mu
        X_test_c  = X_test  - mu
    else:
        X_train_c = X_train
        X_test_c  = X_test

    #%% STEP 5 — PCA for Dimensionality Reduction

    pca = PCA(n_components=NUM_PCS)
    X_train_pca = pca.fit_transform(X_train_c)   # projette les images train sur ses c.p.
    X_test_pca  = pca.transform(X_test_c)        # projette les images test sur les c.p. du train

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
    }
