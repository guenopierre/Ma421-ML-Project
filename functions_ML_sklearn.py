from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

#SVM
svm_sklearn = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr') # librairie sklearn.svm.SVC

#Neural Network
mlp_sklearn = MLPClassifier(
    hidden_layer_sizes=(128,64),  
    activation='logistic',
    solver='adam',
    max_iter=1000,
    random_state=42, 
    verbose=True
)


