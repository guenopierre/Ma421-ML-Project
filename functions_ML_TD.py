import numpy as np
import utils


def predict(Theta1, Theta2, X):    
  
    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    
    # useful variables
    m = X.shape[0] 
    num_labels = Theta2.shape[0]

    # You need to return the following variables correctly 
    p = np.zeros(X.shape[0])    #we have to return the same number of numbers 

    

    # Layer 1 -> Layer 2: add bias unit, compute hidden activations
    X_bias = np.column_stack([np.ones(m), X])          # (m, 401)
    a2 = utils.sigmoid(X_bias @ Theta1.T)              # (m, 25)

    # Layer 2 -> Output: add bias unit, compute output activations
    a2_bias = np.column_stack([np.ones(m), a2])        # (m, 26)
    a3 = utils.sigmoid(a2_bias @ Theta2.T)             # (m, 10)

    # Predicted class is the index of the highest output neuron
    p = np.argmax(a3, axis=1)

    # =============================================================
    return