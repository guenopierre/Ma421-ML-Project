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

def load_images(img_ids, image_dir, img_size):
    """Load all images, preprocess them with OpenCV, flatten to 1D vectors.
    Returns a matrix X of shape (n_samples, img_size*img_size)."""
    n = len(img_ids)
    X = np.zeros((n, img_size * img_size))
    for i, img_id in enumerate(img_ids):
        path = os.path.join(image_dir, img_id + '.jpg')
        img  = preprocess_image(path, IMG_SIZE=(img_size, img_size))
        arr  = img.astype(np.float64) / 255.0       # normalize to [0, 1]
        X[i, :] = arr.flatten()                      # flatten the 2D grayscale image
        if (i + 1) % 500 == 0:
            print(f"    Loaded {i+1}/{n} images...")
    return X