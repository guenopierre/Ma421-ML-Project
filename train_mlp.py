import numpy as np
from scipy import optimize

# 1. Fonctions de base du cours
def sigmoid(z):
    # On limite z pour éviter l'overflow (exp(500) est trop grand)
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_gradient(z):
    g = sigmoid(z)
    return g * (1 - g)

# 2. La fonction de coût "unrolled" (exactement comme dans Exercices_3.ipynb)
def nn_cost_function(nn_params, layer_sizes, X, y, lambda_):
    m = X.shape[0]
    num_labels = layer_sizes[-1]
    
    # Reconstitution des matrices Theta (Rolling)
    thetas = []
    idx = 0
    for i in range(len(layer_sizes) - 1):
        in_s = layer_sizes[i]
        out_s = layer_sizes[i+1]
        size = out_s * (in_s + 1)
        thetas.append(nn_params[idx:idx+size].reshape(out_s, in_s + 1))
        idx += size

    # --- PARTIE 1 : Feedforward ---
    a = [np.concatenate([np.ones((m, 1)), X], axis=1)] # Entrée avec biais
    zs = []
    
    for i in range(len(thetas)):
        z = np.dot(a[i], thetas[i].T)
        zs.append(z)
        a_next = sigmoid(z)
        if i < len(thetas) - 1: # On ajoute le biais sauf pour la couche de sortie
            a_next = np.concatenate([np.ones((m, 1)), a_next], axis=1)
        a.append(a_next)
    
    h = a[-1] # Prédictions finales

    # Codage "one-hot" des labels (y_matrix)
    y_matrix = np.eye(num_labels)[y]

    # Calcul du coût J (Log-Loss)
    # On ajoute 1e-15 pour éviter log(0) qui donne NaN
    term1 = -y_matrix * np.log(h + 1e-15)
    term2 = (1 - y_matrix) * np.log(1 - h + 1e-15)
    J = np.sum(term1 - term2) / m
    
    # Ajout de la régularisation (on ne régularise pas le biais : colonne 0)
    reg = 0
    for t in thetas:
        reg += np.sum(np.square(t[:, 1:]))
    J += (lambda_ / (2 * m)) * reg

    # --- PARTIE 2 : Backpropagation ---
    deltas = [h - y_matrix] # Erreur en sortie
    
    # Remontée des couches
    for i in range(len(thetas) - 1, 0, -1):
        d = np.dot(deltas[0], thetas[i][:, 1:]) * sigmoid_gradient(zs[i-1])
        deltas.insert(0, d)
    
    # Calcul des gradients pour chaque Theta
    grads = []
    for i in range(len(thetas)):
        grad = np.dot(deltas[i].T, a[i]) / m
        grad[:, 1:] += (lambda_ / m) * thetas[i][:, 1:]
        grads.append(grad.ravel())
        
    return J, np.concatenate(grads)

# 3. Classe Wrapper pour compatibilité avec ton evaluate.py
class HandmadeMLP:
    def __init__(self, params, layer_sizes, classes):
        self.params = params
        self.layer_sizes = layer_sizes
        self.classes_ = classes

    def predict_proba(self, X):
        """ Calcule les activations de la dernière couche (probabilités) """
        a = X
        idx = 0
        thetas = []
        # On extrait les matrices Theta des paramètres mis à plat
        for i in range(len(self.layer_sizes) - 1):
            size = self.layer_sizes[i+1] * (self.layer_sizes[i] + 1)
            thetas.append(self.params[idx:idx+size].reshape(self.layer_sizes[i+1], self.layer_sizes[i] + 1))
            idx += size
            
        # Propagation avant (Feedforward)
        for i, t in enumerate(thetas):
            # Ajout du biais (colonne de 1)
            a = np.concatenate([np.ones((a.shape[0], 1)), a], axis=1)
            # Calcul de l'activation
            a = sigmoid(np.dot(a, t.T))
        
        return a

    def predict(self, X):
        """ Retourne la classe avec la probabilité la plus haute """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

# 4. Fonction d'entraînement principale
def train_mlp(X_train, y_train, hidden_layers=(1024, 512, 256), max_iter=150, 
              alpha=1e-4, random_state=42, verbose=True):
    
    np.random.seed(random_state)
    classes = np.unique(y_train)
    input_size = X_train.shape[1]
    num_labels = len(classes)
    layer_sizes = [input_size] + list(hidden_layers) + [num_labels]
    
    # Initialisation aléatoire (Casser la symétrie)
    nn_params = []
    for i in range(len(layer_sizes)-1):
        epsilon = 0.12 # Valeur type du cours
        w = np.random.rand(layer_sizes[i+1], 1 + layer_sizes[i]) * 2 * epsilon - epsilon
        nn_params.append(w.ravel())
    initial_params = np.concatenate(nn_params)
    
    # Utilisation de l'optimiseur Scipy (beaucoup plus puissant qu'une boucle manuelle)
    res = optimize.minimize(fun=nn_cost_function,
                            x0=initial_params,
                            args=(layer_sizes, X_train, y_train, alpha),
                            method='L-BFGS-B', # Recommandé pour les gros vecteurs de paramètres
                            jac=True,
                            options={'maxiter': max_iter, 'disp': verbose})

    return HandmadeMLP(res.x, layer_sizes, classes)