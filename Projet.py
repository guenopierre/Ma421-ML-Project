import time
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from functions_ML_TD import *
from functions_ML_sklearn import *
from confusion_matrix_display import *


#%% récupération des données prétraitées (dans le fichier preprocess.py)

import preprocess 

X_train_pca = preprocess.X_train_pca
y_train = preprocess.y_train

X_test_pca = preprocess.X_test_pca
y_test = preprocess.y_test

classes     = preprocess.classes
n_train     = preprocess.n_train
n_test      = preprocess.n_test
num_classes = preprocess.num_classes

IMG_SIZE = preprocess.IMG_SIZE
NUM_PCS  = preprocess.NUM_PCS

#%% SVM (Training)

start_time = time.time()
svm_sklearn.fit(X_train_pca, y_train)  #file: functions_ML_sklearn
elapsed = time.time() - start_time
print(f"  Training complete in {elapsed:.2f} seconds.")
 
#%% SVM (Test)

print("Start of the SVM test")
start_time = time.time()

y_pred_test_SVM  = svm_sklearn.predict(X_test_pca) #file: functions_ML_sklearn
test_acc_SVM  = accuracy_score(y_test,  y_pred_test_SVM)

elapsed = time.time() - start_time

print(f"  SVM Test complete in {elapsed:.2f} seconds.")
print(f"  Test accuracy     : {test_acc_SVM:.4f}  ({test_acc_SVM*100:.1f}%) ")

#%% SVM (Test) - Confusion Matrix

cm_SVM = confusion_matrix(y_test, y_pred_test_SVM)
confusion_matrix_display(cm_SVM, y_test, num_classes, classes, f"Matrice de confusion (normalisée) / SVM / {NUM_PCS} composantes principales / images de taille : {IMG_SIZE} x {IMG_SIZE} ", save = False)  #file: confusion_matrix_display

#%% Neural Network - MLP (Training)

start_time = time.time()
mlp_sklearn.fit(X_train_pca, y_train) #file: functions_ML_sklearn
elapsed = time.time() - start_time
print(f"  MLP training complete in {elapsed:.2f} seconds.")

#%% Neural Network - MLP (Test)

start_time = time.time()

y_pred_test_mlp  = mlp_sklearn.predict(X_test_pca) #file: functions_ML_sklearn
test_acc_mlp  = accuracy_score(y_test,  y_pred_test_mlp)

elapsed = time.time() - start_time

print(f"  MLP Test complete in {elapsed:.2f} seconds.")
print(f"  Test accuracy     : {test_acc_mlp:.4f}  ({test_acc_mlp*100:.1f}%)")

#%% Neural Network - MLP (Test) - Details

print("\nClassification Report (MLP — Test Set):")
print(classification_report(y_test, y_pred_test_mlp, target_names=classes))

#%% Neural Network - MLP (Test) - Confusion Matrix

cm_mlp = confusion_matrix(y_test, y_pred_test_mlp)
confusion_matrix_display(cm_mlp, y_test, num_classes, classes, f"Matrice de confusion (normalisée) / MLP / {NUM_PCS} composantes principales / images de taille : {IMG_SIZE} x {IMG_SIZE} ", save = False)  #file: confusion_matrix_display


