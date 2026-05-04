from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


#%%SVM
svm_sklearn = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr') # librairie sklearn.svm.SVC

#%%Neural Network

#d'après la boucle ci dessous, les paramètres optimaux sont ceux-ci:
    
hidden_layers = (1024,512)
max_iter = 5000

mlp_sklearn = MLPClassifier(
    hidden_layer_sizes=hidden_layers,  
    activation='relu',
    solver='adam',
    max_iter=max_iter,
    random_state=42, 
    verbose=False
)


#%%MLP - Recherche des meilleurs hyperparamètres (RandomizedSearchCV)


# #Meilleurs paramètres  : {'hidden_layer_sizes': (1024, 512), 'activation': 'relu'}
# #Meilleure accuracy CV : 0.1563  (15.6%)
# #Accuracy sur test set : 0.1932  (19.3%)

# sizes = [64, 128, 256, 512, 1024]

# architectures = (
#     [(n,)             for n in sizes] +
#     [(n1, n2)         for n1 in sizes for n2 in sizes if n1 >= n2] +
#     [(n1, n2, n3)     for n1 in sizes for n2 in sizes for n3 in sizes if n1 >= n2 >= n3] +
#     [(n1, n2, n3, n4) for n1 in sizes for n2 in sizes for n3 in sizes for n4 in sizes if n1 >= n2 >= n3 >= n4]
# )
# print(f"Espace de recherche : {len(architectures)} architectures possibles")

# param_dist = {
#     'hidden_layer_sizes': architectures,
#     'activation': ['relu', 'logistic', 'tanh'],
# }

# mlp_base = MLPClassifier(solver='adam', max_iter=1000, random_state=42)

# search = RandomizedSearchCV(
#     mlp_base,
#     param_distributions=param_dist,
#     n_iter=200,       # nombre de combinaisons testées au hasard 
#     cv=3,            # cross-validation en 3 plis
#     scoring='accuracy',
#     random_state=42,
#     verbose=2,
#     n_jobs=-1        # utilise tous les coeurs CPU
# )

# start_time = time.time()
# search.fit(X_train_pca, y_train)
# elapsed = time.time() - start_time
# print(f"\nRecherche terminée en {elapsed:.1f} secondes.")

# print(f"Meilleurs paramètres  : {search.best_params_}")
# print(f"Meilleure accuracy CV : {search.best_score_:.4f}  ({search.best_score_*100:.1f}%)")

# y_pred_best   = search.predict(X_test_pca)
# test_acc_best = accuracy_score(y_test, y_pred_best)
# print(f"Accuracy sur test set : {test_acc_best:.4f}  ({test_acc_best*100:.1f}%)")


