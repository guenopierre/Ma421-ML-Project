import cv2
import numpy as np
import os


# =============================================================================
#  Fonctions COMMUNES (identiques pour N&B et couleur)
# =============================================================================

def load_annotations(filepath):
    """Read an FGVC-Aircraft annotation file.
    Format per line: '<7-char image_id> <family label>'
    Returns lists of image IDs and string labels."""
    img_ids, labels = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id = line[:7]       # first 7 characters = image ID
            label  = line[8:]       # everything after the space = family name
            img_ids.append(img_id)
            labels.append(label)
    return img_ids, labels


def balance_classes(img_ids, labels, target_count=None, random_state=42):
    """
    Sous-échantillonne chaque classe pour qu'elles aient toutes le même nombre d'images.
    Cela évite que le modèle privilégie les classes majoritaires.

    Parameters
    ----------
    img_ids : list of str
        Liste des identifiants d'images.
    labels : list of str (or array)
        Liste des labels (même longueur que img_ids).
    target_count : int or None
        Nombre d'images à garder par classe.
        Si None, prend la taille de la classe la plus petite (downsampling complet).
    random_state : int
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    img_ids_bal : list of str
        Liste équilibrée d'identifiants d'images.
    labels_bal : list
        Liste équilibrée de labels (même type que `labels` en entrée).
    """
    rng = np.random.default_rng(random_state)

    # On convertit en array numpy pour faciliter l'indexation
    img_ids = np.array(img_ids)
    labels  = np.array(labels)

    # Compte le nombre d'images par classe
    unique_classes, counts = np.unique(labels, return_counts=True)

    # Si aucune cible n'est donnée, on prend la classe la plus petite
    if target_count is None:
        target_count = counts.min()

    print(f"  Balancing classes: {len(unique_classes)} classes, "
          f"target = {target_count} images/class")

    selected_indices = []
    for cls in unique_classes:
        # Indices des images appartenant à cette classe
        cls_indices = np.where(labels == cls)[0]
        n_available = len(cls_indices)

        if n_available >= target_count:
            # Sous-échantillonnage sans remise
            chosen = rng.choice(cls_indices, size=target_count, replace=False)
        else:
            # Si la classe a moins d'images que la cible -> on prend tout
            # (alternative : sur-échantillonner avec remise, mais ce n'est pas
            #  recommandé ici car cela duplique des images)
            print(f"    /!\\ Class '{cls}' has only {n_available} images "
                  f"(< target {target_count}), keeping all.")
            chosen = cls_indices

        selected_indices.extend(chosen.tolist())

    # On mélange l'ordre final pour ne pas avoir toutes les classes regroupées
    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    img_ids_bal = img_ids[selected_indices].tolist()
    labels_bal  = labels[selected_indices].tolist()

    print(f"  Balanced dataset size: {len(img_ids_bal)} images "
          f"(was {len(img_ids)})")

    return img_ids_bal, labels_bal


# =============================================================================
#  Fonctions de PRÉPROCESSING D'UNE IMAGE
#   - version GRAY  : N&B + crop du bandeau de crédit photo (20px du bas)
#   - version COLOR : RGB façon Matlab, sans crop
# =============================================================================

def preprocess_image_gray(image_path, IMG_SIZE=(64, 64)):
    """
    Loads an image, converts it to grayscale, removes the bottom 20 pixels
    (photo credit banner), and resizes it to IMG_SIZE.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    # Remove bottom 20 pixels (credit banner)
    height, width = img.shape
    if height <= 20:
        raise ValueError("Image too small to crop 20 pixels from the bottom.")
    img_cropped = img[:-20, :]

    # Resize to IMG_SIZE
    img_resized = cv2.resize(img_cropped, IMG_SIZE)

    return img_resized


def preprocess_image_color(image_path, IMG_SIZE=(64, 64)):
    """
    Replicates the Matlab preprocessing exactly:
      1. Load image in COLOR (3 channels, RGB)
      2. Resize directly to IMG_SIZE  (no bottom-banner cropping:
         the Matlab code relies on the resize to make the banner negligible)
      3. Return the resized RGB image as a (H, W, 3) array
    """
    # Load the image in color. cv2 uses BGR by default, so we convert to RGB
    # to match Matlab's imread() which returns RGB.
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize directly to IMG_SIZE (NO cropping — same as Matlab)
    img_resized = cv2.resize(img, IMG_SIZE)

    return img_resized


# =============================================================================
#  Fonctions de CHARGEMENT D'UN BATCH D'IMAGES
#   - version GRAY  : retourne une matrice (n, IMG_SIZE*IMG_SIZE)
#   - version COLOR : retourne une matrice (n, IMG_SIZE*IMG_SIZE*3) en ordre Fortran
# =============================================================================

def load_images_gray(img_ids, image_dir, img_size):
    """Load all images, preprocess them with OpenCV (grayscale), flatten to 1D vectors.
    Returns a matrix X of shape (n_samples, img_size*img_size)."""
    n = len(img_ids)
    X = np.zeros((n, img_size * img_size))
    for i, img_id in enumerate(img_ids):
        path = os.path.join(image_dir, img_id + '.jpg')
        img  = preprocess_image_gray(path, IMG_SIZE=(img_size, img_size))
        arr  = img.astype(np.float64) / 255.0       # normalize to [0, 1]
        X[i, :] = arr.flatten()                      # flatten the 2D grayscale image
        if (i + 1) % 500 == 0:
            print(f"    Loaded {i+1}/{n} images...")
    return X


def load_images_color(img_ids, image_dir, img_size):
    """Load all images, preprocess them (resize to img_size x img_size in RGB),
    flatten each to a 1D vector and normalize to [0, 1].

    To stay perfectly aligned with the Matlab pipeline, the flattening uses
    Fortran (column-major) order, which is Matlab's native memory layout.
    Each image therefore produces a vector of size img_size*img_size*3.

    Returns a matrix X of shape (n_samples, img_size*img_size*3).
    """
    n = len(img_ids)
    n_features = img_size * img_size * 3
    X = np.zeros((n, n_features))
    for i, img_id in enumerate(img_ids):
        path = os.path.join(image_dir, img_id + '.jpg')
        img  = preprocess_image_color(path, IMG_SIZE=(img_size, img_size))
        arr  = img.astype(np.float64) / 255.0           # normalize to [0, 1]
        # Fortran order matches Matlab's img(:) column-major flattening.
        X[i, :] = arr.flatten(order='F')
        if (i + 1) % 500 == 0:
            print(f"    Loaded {i+1}/{n} images...")
    return X
