#%% LIBRAIRIES 

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
 
IMG_SIZE = 128      # resize images to 64x64 (same as MATLAB: imgSize = 64)
NUM_PCS  = 500    # number of principal components to keep

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

pca = PCA(n_components=NUM_PCS)
X_train_pca = pca.fit_transform(X_train)   # projette les images train sur ses composantes principales 
X_test_pca  = pca.transform(X_test)        # projette les images test sur les c.p. de l'ensemble train 


#%% # STEP 5 — 1st Classification Test with SVM --> only to calibrate the best NUM_PCS

#utilisation de la librairie sklearn aussi (fct SVC)

svm_clf = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')

# SVC parameters:
#   C     = regularisation parameter (equivalent to 'C' in the course, eq. 97)
#   kernel = 'rbf' → Gaussian kernel: K(x,l) = exp(-||x-l||² / 2σ²)
#   gamma  = 1/(2σ²), controls the smoothness of the decision boundary

start_time = time.time()
svm_clf.fit(X_train_pca, y_train)
elapsed = time.time() - start_time
print(f"  Training complete in {elapsed:.2f} seconds.")
 
#%% STEP 6 — Prediction and Evaluation

y_pred_train = svm_clf.predict(X_train_pca)
y_pred_test  = svm_clf.predict(X_test_pca)
 
train_acc = accuracy_score(y_train, y_pred_train)
test_acc  = accuracy_score(y_test,  y_pred_test)

#%% affichage des résultats

print(f"  Training accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")
print(f"  Test accuracy    : {test_acc:.4f}  ({test_acc*100:.1f}%)")
 
# Detailed classification report (precision, recall, F1 per class)
print("\n--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred_test, target_names=classes))

# Confusion matrix (colors only, no numbers)
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix — Aircraft Family Classification')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, classes, rotation=45, ha='right', fontsize=8)
plt.yticks(tick_marks, classes, fontsize=8)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'confusion_matrix.png'), dpi=150)
plt.show()
print("  → Figure saved: confusion_matrix.png")
 


#%% BIS — Loop: Training Accuracy vs NUM_PCS

pcs_range = list(range(100, 3334 + 1, 100))
test_accuracies = []

# PCA complète une seule fois
print("\nFitting full PCA once...")
pca_full = PCA(n_components=max(pcs_range))
X_train_pca_full = pca_full.fit_transform(X_train)
X_test_pca_full  = pca_full.transform(X_test)

for n_pcs in pcs_range:
    X_train_reduced = X_train_pca_full[:, :n_pcs]
    X_test_reduced  = X_test_pca_full[:, :n_pcs]
    
    svm_temp = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr')
    svm_temp.fit(X_train_reduced, y_train)
    
    y_pred = svm_temp.predict(X_test_reduced)
    acc = accuracy_score(y_test, y_pred)
    test_accuracies.append(acc)
    
    print(f"  NUM_PCS = {n_pcs:4d}  →  test_acc = {acc:.4f} ({acc*100:.1f}%)")

# Tracé du graphe
plt.figure(figsize=(10, 5))
plt.plot(pcs_range, test_accuracies, 'r-o', markersize=4)
plt.xlabel('Number of Principal Components (NUM_PCS)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Number of Principal Components')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.tight_layout()
plt.savefig(os.path.join(DATA_PATH, 'accuracy_vs_num_pcs.png'), dpi=150)
plt.show()
print("  → Figure saved: accuracy_vs_num_pcs.png")
 
