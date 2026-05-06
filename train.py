"""
train.py
─────────────────────────────────────────────────────────────────────────────
Entraînement du réseau de neurones MLP.

Architecture : (1024, 512, 256)
  - Activation ReLU
  - Optimiseur Adam, learning rate 3e-4
  - Régularisation L2 (alpha=1e-4)
  - Early stopping DÉSACTIVÉ → le MLP converge jusqu'à max_iter
    Raison : avec 70 classes et train augmenté, le score de validation
    interne (~20%) est sous-estimé par rapport au vrai score test (~31%).
    Le early stopping déclenchait un arrêt prématuré à 43-159 itérations.
─────────────────────────────────────────────────────────────────────────────
"""

from sklearn.neural_network import MLPClassifier


def train_mlp(X_train, y_train,
              hidden_layers        = (1024, 512, 256),
              activation           = 'relu',
              solver               = 'adam',
              max_iter             = 500,
              learning_rate_init   = 3e-4,
              alpha                = 1e-4,
              early_stopping       = False,
              validation_fraction  = 0.10,
              n_iter_no_change     = 50,
              tol                  = 1e-5,
              random_state         = 42,
              verbose              = True):
    """
    Entraîne un MLPClassifier sklearn.

    Paramètres clés
    ---------------
    early_stopping=False, max_iter=300
        Le early stopping donnait un score de validation interne ~20%
        alors que le vrai score test était ~31% → arrêt prématuré systématique.
        On laisse le MLP converger sur les 300 epochs complets.

    alpha=1e-4 (L2)
        Régularisation pour éviter le surapprentissage sur le train augmenté
        (qui contient deux versions très proches de chaque image).

    learning_rate_init=3e-4
        Plus faible que le défaut (1e-3) : convergence plus stable et fine.
    """
    print(f"  Architecture : {hidden_layers}")
    print(f"  Activation   : {activation}  |  Optimiseur : {solver}")
    print(f"  LR init      : {learning_rate_init}  |  L2 alpha : {alpha}")
    print(f"  Max epochs   : {max_iter}  |  Early stopping : {early_stopping}")

    mlp = MLPClassifier(
        hidden_layer_sizes   = hidden_layers,
        activation           = activation,
        solver               = solver,
        max_iter             = max_iter,
        learning_rate_init   = learning_rate_init,
        alpha                = alpha,
        early_stopping       = early_stopping,
        validation_fraction  = validation_fraction,
        n_iter_no_change     = n_iter_no_change,
        tol                  = tol,
        random_state         = random_state,
        verbose              = verbose,
    )

    mlp.fit(X_train, y_train)

    print(f"\n  Convergé en {mlp.n_iter_} itérations.")
    print(f"  Loss finale  : {mlp.loss_:.6f}")

    return mlp