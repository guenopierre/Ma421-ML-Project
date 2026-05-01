from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


#%%SVM
svm_sklearn = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr') # librairie sklearn.svm.SVC

#%%Neural Network

hidden_layers = (512,512,512,512)
max_iter = 5000

mlp_sklearn = MLPClassifier(
    hidden_layer_sizes=hidden_layers,  
    activation='relu',
    solver='adam',
    max_iter=max_iter,
    random_state=42, 
    verbose=False
)


