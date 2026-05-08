"""
train_svm.py
─────────────────────────────────────────────────────────────────────────────
Entraînement d'un SVM (Support Vector Machine) avec noyau RBF.

Pourquoi un SVM en complément du MLP ?
  - Le SVM cherche la frontière de décision à marge maximale → excellent
    quand les classes sont bien séparées dans l'espace PCA
  - Souvent plus robuste que le MLP sur des datasets de taille moyenne
    (~5000-10000 images)
  - Complémentaire : là où le MLP hésite, le SVM peut être plus tranché,
    et inversement

Paramètres choisis :
  kernel='rbf'   → noyau gaussien, standard pour la vision
  C=10           → pénalise fortement les erreurs (bon avec PCA propre)
  gamma='scale'  → gamma = 1 / (n_features * X.var()), auto-adaptatif
  probability=True → active predict_proba() pour le rapport de classif
─────────────────────────────────────────────────────────────────────────────
"""

from sklearn.svm import SVC


def train_svm(X_train, y_train,
              kernel      = 'rbf',
              C           = 15,
              gamma       = 'scale',
              probability = True,
              random_state = 42,
              verbose      = True):
    """
    Entraîne un SVM multiclasse (stratégie one-vs-one par défaut dans sklearn).

    Paramètres
    ----------
    C : float
        Paramètre de régularisation. Plus C est grand, plus le SVM essaie
        de classer tous les exemples d'entraînement correctement
        (moins de marge, plus de risque de surapprentissage).
        C=10 est un bon compromis après PCA.

    gamma : 'scale' | 'auto' | float
        Coefficient du noyau RBF. 'scale' = 1/(n_features * var(X)),
        s'adapte automatiquement à la dimension PCA.

    probability : bool
        Active le calcul des probabilités par calibration Platt.
        Légèrement plus lent à entraîner mais permet predict_proba().

    Retourne
    --------
    svm : SVC entraîné
    """
    if verbose:
        print(f"  Kernel : {kernel}  |  C={C}  |  gamma={gamma}")
        print(f"  Multiclasse : one-vs-one  |  probability={probability}")
        print(f"  (Le SVM peut prendre plusieurs minutes sur ~9000 images...)")

    svm = SVC(
        kernel       = kernel,
        C            = C,
        gamma        = gamma,
        probability  = probability,
        decision_function_shape = 'ovr',
        random_state = random_state,
    )

    svm.fit(X_train, y_train)

    if verbose:
        print(f"  SVM entraîné — {len(svm.support_vectors_)} vecteurs supports")

    return svm