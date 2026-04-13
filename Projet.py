#%% LIBRAIRIES 

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from functions import *


#%% PREMIERE SEANCE ENSEMBLE (LIVESHARE)

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


#%% STEP 1 — Configuration
# /!\ chemin RELATIF /!\ il faut run en précisant qu'on est dans Ma421-ML-Project folder
#sinon indiquer le chemin absolue mais dépend de l'ordi de chacun

DATA_PATH = '.'

TRAIN_ANNOT = os.path.join(DATA_PATH, 'images_family_train.txt')
TEST_ANNOT  = os.path.join(DATA_PATH, 'images_family_test.txt')
IMAGE_DIR   = os.path.join(DATA_PATH, 'images')
 
IMG_SIZE = 64      # resize images to 64x64 (same as MATLAB: imgSize = 64)
NUM_PCS  = 500      # number of principal components to keep (same as MATLAB: numPCs = 100)

#%% STEP 2 — Load Annotations

train_imgs, labels_train = load_annotations(TRAIN_ANNOT)
test_imgs,  labels_test  = load_annotations(TEST_ANNOT)

# convertir le label (ex: 'A330') en entier (ex: 3) 
#il le fait par ordre alphabétique
le = LabelEncoder()
le.fit(labels_train + labels_test)
y_train = le.transform(labels_train)
y_test  = le.transform(labels_test)

classes     = le.classes_
n_train     = len(train_imgs)
n_test      = len(test_imgs)
num_classes = len(classes)

#%% STEP 3 — Feature Extraction (resize + flatten + normalize)
# Preprocessing with OpenCV (cv2):
#   1. Load image in grayscale (cv2.IMREAD_GRAYSCALE)
#   2. Remove the bottom 20 pixels (photo credit banner, useless for the algo)
#   3. Resize to IMG_SIZE x IMG_SIZE
#   4. Flatten to a 1D vector and normalize to [0, 1]
#
# Since we work in grayscale, each image produces a vector of size IMG_SIZE*IMG_SIZE
# (instead of IMG_SIZE*IMG_SIZE*3 for RGB).

# Pour les grosses matrices X
#chaque ligne = 1 photo (car en N&B on peut voir la photo comme un vecteur)
#chaque colonne = 1 pixel de notre photo en nuance de gris (ici 4096 car on les a redimensionner en 64*64 = 4096)
print("\nLoading training images...")
X_train = load_images(train_imgs, IMAGE_DIR, IMG_SIZE)
 
print("Loading test images...")
X_test = load_images(test_imgs, IMAGE_DIR, IMG_SIZE)
 
print(f"  X_train shape: {X_train.shape}  (n_samples × n_features)")
print(f"  X_test  shape: {X_test.shape}")


#%% STEP 4 — PCA for Dimensionality Reduction

#j'ai fais le choix d'utiliser sklearn directement
#si on a du temps, on pourrait faire notre ACP nous même en prenant les cours de Couffi

# In sklearn, PCA automatically:
#   - centers the data (subtracts the mean internally)
#   - computes the SVD to find principal components
#   - projects onto the top n_components
#
# sklearn.decomposition.PCA parameters:
#   n_components = number of principal components K to keep
#   The attribute components_ corresponds to MATLAB's coeff (eigenvectors)
#   The method transform() projects the data: Z = (X - mu) @ coeff[:, :K]
 
print(f"\nApplying PCA with n_components = {NUM_PCS}...")
 
pca = PCA(n_components=NUM_PCS)
X_train_pca = pca.fit_transform(X_train)   # fit on training data + project
X_test_pca  = pca.transform(X_test)         # project test data (same basis)
 
print(f"  X_train after PCA: {X_train_pca.shape}  (reduced from {X_train.shape[1]} to {NUM_PCS})")
print(f"  X_test  after PCA: {X_test_pca.shape}")
 
# --- Explained variance analysis ---
# This corresponds to the course metric (eq. 105):
#   1 - sum(S_j, j=1..K) / sum(S_j, j=1..M)  <=  epsilon
# where S_j are the eigenvalues. sklearn gives this as explained_variance_ratio_.
 
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
total_variance_kept = cumulative_variance[-1]
print(f"  Cumulative variance retained with {NUM_PCS} PCs: {total_variance_kept:.4f} ({total_variance_kept*100:.1f}%)")
 
# Plot the cumulative explained variance
plt.figure(figsize=(8, 4))
plt.plot(range(1, NUM_PCS + 1), cumulative_variance, 'b-o', markersize=2)
plt.xlabel('Number of Principal Components (K)')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA — Choosing K (number of components)')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'pca_variance.png'), dpi=150)
plt.show()
print("  → Figure saved: pca_variance.png")
 
 
