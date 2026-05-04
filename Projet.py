import time
import matplotlib.pyplot as plt
import importlib, functions_ML_sklearn

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from functions_ML_TD import *
from functions_ML_sklearn import *
from confusion_matrix_display import *



#%% PREPROCESS 


# 0 = N&B; 1 = RGB
images_couleur = 1
 
# 0 = pas d'équilibrage des classes; 1 = sous-échantillonnage pour avoir le même nombre d'images par classe
balance_tri = 0 #pour le train
balance_test = 1 #pour le test
 
# redimensions de nos images (carrées)
IMG_SIZE = 64

# composantes principales gardées
NUM_PCS  = 100
 
from preprocess import run_preprocessing
 
pp = run_preprocessing(
    images_couleur = images_couleur,
    balance_tri    = balance_tri,
    balance_test   = balance_test,
    IMG_SIZE       = IMG_SIZE,
    NUM_PCS        = NUM_PCS,
)
 
 
X_train_pca = pp['X_train_pca']
y_train     = pp['y_train']
 
X_test_pca  = pp['X_test_pca']
y_test      = pp['y_test']
 
classes     = pp['classes']
n_train     = pp['n_train']
n_test      = pp['n_test']
num_classes = pp['num_classes']
 
IMG_SIZE    = pp['IMG_SIZE']     
NUM_PCS     = pp['NUM_PCS']
COLOR_SIZE  = pp['COLOR_SIZE']



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
confusion_matrix_display(cm_SVM, y_test, num_classes, classes, f"Matrice de confusion (normalisée) / SVM / {NUM_PCS} composantes principales / images de taille : {COLOR_SIZE} x {IMG_SIZE} x {IMG_SIZE} ", save = False)  #file: confusion_matrix_display

#%% Neural Network - MLP (Training)

importlib.reload(functions_ML_sklearn)

print(f"Hidden Layers: ({hidden_layers})\nMax iterations: {max_iter}")

start_time = time.time()
print("MLP running...")
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
confusion_matrix_display(cm_mlp, y_test, num_classes, classes, f"Matrice de confusion (normalisée) / MLP / accuracy: {test_acc_mlp*100:.1f}% \n hidden layers: {hidden_layers} / {NUM_PCS} composantes principales / images de taille : {COLOR_SIZE} x {IMG_SIZE} x {IMG_SIZE} ", save = False)  #file: confusion_matrix_display



