#%% LIBRAIRIES 

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


from functions_preprocess import *
from functions_ML_TD import *


#%% STEP 1 — Configuration
# /!\ chemin RELATIF /!\ il faut run en précisant qu'on est dans Ma421-ML-Project folder
#sinon indiquer le chemin absolue mais dépend de l'ordi de chacun

DATA_PATH = '.'

TRAIN_ANNOT = os.path.join(DATA_PATH, 'images_family_train.txt')
TEST_ANNOT  = os.path.join(DATA_PATH, 'images_family_test.txt')
IMAGE_DIR   = os.path.join(DATA_PATH, 'images')
 
IMG_SIZE = 128      # resize images in a square (IMG_SIZE * IMG_SIZE)
NUM_PCS  = 100    # number of principal components to keep

#%% STEP 2 — Load Annotations

train_imgs, labels_train = load_annotations(TRAIN_ANNOT)
test_imgs,  labels_test  = load_annotations(TEST_ANNOT)

# convertir le label (ex: 'A330') en entier (ex: 3) 
#--> il le fait par ordre alphabétique
le = LabelEncoder()
le.fit(labels_train + labels_test)
y_train = le.transform(labels_train)
y_test  = le.transform(labels_test)

classes     = le.classes_
n_train     = len(train_imgs)
n_test      = len(test_imgs)
num_classes = len(classes)

#%% STEP 3 — Feature Extraction (resize + flatten + normalize)

# see the function_preprocess.py file 

# Preprocessing with OpenCV (cv2):
#   1. Load image in grayscale (cv2.IMREAD_GRAYSCALE)
#   2. Remove the bottom 20 pixels (photo credit banner, useless for the algo)
#   3. Resize to IMG_SIZE x IMG_SIZE  --> on pourrait s'intéresser à garder le même ration L/l pour nos images
#   4. Flatten to a 1D vector and normalize to [0, 1]
#
# Since we work in grayscale, each image produces a vector of size IMG_SIZE*IMG_SIZE
# (instead of IMG_SIZE*IMG_SIZE*3 for RGB).

# Pour les grosses matrices X
#chaque ligne = 1 photo (car en N&B on peut voir la photo comme un vecteur)
#chaque colonne = 1 pixel de notre photo en nuance de gris /!\ oui nos images sont mis sous formes de vecteur
print("\nLoading training images...")
X_train = load_images(train_imgs, IMAGE_DIR, IMG_SIZE)
 
print("Loading test images...")
X_test = load_images(test_imgs, IMAGE_DIR, IMG_SIZE)
 
print(f"  X_train shape: {X_train.shape}  (n_samples × n_features)")
print(f"  X_test  shape: {X_test.shape}")


#%% STEP 4 — PCA for Dimensionality Reduction

#j'ai fais le choix d'utiliser sklearn directement
#si on a du temps, on pourrait faire notre ACP nous même en prenant les cours de Couffi ou celui d'Ortega

pca = PCA(n_components=NUM_PCS)
X_train_pca = pca.fit_transform(X_train)   # projette les images train sur ses composantes principales 
X_test_pca  = pca.transform(X_test)        # projette les images test sur les c.p. de l'ensemble train (faut faire attention à garder les mêmes)