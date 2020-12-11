
import numpy as np
import HelperFunctions as hp


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation 

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shpe[1]

    # FORWARD PROPAGATION
    A = hp.sigmoid(np.dot(w.T, X) + b) # Compute actcivation
    cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A))) #compute cost
    
    # BACKWARD PROPAGATION
    dw = (1/m) * np.dot(X, (A - Y).T)
    db = (1/m) * np.sum(A - Y)

    # ASSERTS
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw, "db": db}
    
    return grads, cost
    