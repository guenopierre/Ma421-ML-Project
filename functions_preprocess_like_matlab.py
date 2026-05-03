import cv2
import numpy as np
import os


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


def preprocess_image(image_path, IMG_SIZE=(64, 64)):
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


def load_images(img_ids, image_dir, img_size):
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
        img  = preprocess_image(path, IMG_SIZE=(img_size, img_size))
        arr  = img.astype(np.float64) / 255.0           # normalize to [0, 1]
        # Fortran order matches Matlab's img(:) column-major flattening.
        X[i, :] = arr.flatten(order='F')
        if (i + 1) % 500 == 0:
            print(f"    Loaded {i+1}/{n} images...")
    return X
