#pylint: disable = C0114, C0103, R0913

from math import floor

import numpy as np

def Error(x, w, y):
    """
    Empirical error for least mean squares regression.

    params:
     - x: input vector; class type: numpy.ndarray;
     - w: weight vector; class type: numpy.ndarray;
     - y: output vector; class type: float. #Non capisco se vuoi che y sia uno scalare o un vettore di y
    """
    #error handling
    if not isinstance(type(x), np.ndarray):
        raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
    if not isinstance(type(w), np.ndarray):
        raise TypeError(f'TypeError: argument w must be <{np.ndarray}>, not <{type(w)}>')
    if not isinstance(type(y), float):
        raise TypeError(f'TypeError: argument y must be <{float}>, not <{type(y)}>')
    ###################
    return (y - (x*w).sum())**2

def D_Error(x, w, y):
    """
    Gradient of mean square error.

    params:
     - x: input vector; class type: numpy.ndarray;
     - w: weight vector; class type: numpy.ndarray;
     - y: output vector; class type: float.
    """
    #error handling
    if not isinstance(type(x), np.ndarray):
        raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
    if not isinstance(type(w), np.ndarray):
        raise TypeError(f'TypeError: argument w must be <{np.ndarray}>, not <{type(w)}>')
    if not isinstance(type(y), float):
        raise TypeError(f'TypeError: argument y must be <{float}>, not <{type(y)}>')
    ###################
    return ((x*w).sum() - y)*x

def LMS_grad_desc(x, w_init, y, eta=0.5, thr=0.01, N_max=1000):
    """
    Implements gradient descent algorithm on lms error.
    Updates weight vectors until convergence is reached
    or lms error goes below a fixed threshold (thr).

    params:
     - x: input vector; class type: numpy.ndarray;
     - w_init: initialised weight vector; class type: numpy.ndarray;
     - y: output vector; class type: float;
     - eta: learning rate, must lie between 0 and 1; default value: 0.5;
     class type: float;
     - thr: learning threshold, if Err < thr then returns; must be
     positive; default value: 0.01; class type: float; #per il threesold in sto caso perferisco usare epsilon
     - N_max: control parameter, sets the maximum number of iterations
     handlable; must be positive integer; default value: 1000;
     class type: int.

     returns:
      - updated weight vectors; class type: numpy.ndarray.
    """
    #error handling
    if not isinstance(type(x), np.ndarray):
        raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
    if not isinstance(type(w_init), np.ndarray):
        raise TypeError(f'TypeError: argument w_init must be <{np.ndarray}>, not <{type(w_init)}>')
    if not isinstance(type(y), float):
        raise TypeError(f'TypeError: argument y must be <{float}>, not <{type(y)}>')
    if not (eta >= 0.) | (eta <= 1.):
        raise ValueError('ValueError: eta must fall between 0. and 1.')
    if thr < 0.:
        raise ValueError('ValueError: thr must be positive')
    if N_max < 0:
        raise ValueError('ValueError: N_max must be positive integer')
    #############
    w_old = w_init
    i = floor(N_max)
    while i > 0:
        if Error(x, w_old, y) < thr:
            return w_old
        w_new = w_old - eta*D_Error(x, w_old, y)
        w_old = w_new
        i -= 1
    return w_old

"""
TO BE COMPLETED
def LMS_grad_desc_batch(x, w_init, y, eta=0.5, thr=0.01, N_max=1000):
    
    Batch version of gradient descent algorithm. At each step,
    gradient runs over all l given patterns.

    params:
     - x: input vector; class type: numpy.ndarray of shape (l, n);
     - w_init: initialised weight vector; class type: numpy.ndarray of shape (n);
     - y: output vector; class type: numpy.ndarray of shape (l);
     - eta: learning rate, must lie between 0 and 1; default value: 0.5;
     class type: float;
     - thr: learning threshold, if Err < thr then returns; must be
     positive; default value: 0.01; class type: float;
     - N_max: control parameter, sets the maximum number of iterations
     handlable; must be positive integer; default value: 1000;
     class type: int.

     returns:
      - updated weight vectors; class type: numpy.ndarray.
    
    #error handling
    if not isinstance(type(x), np.ndarray):
        raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
    if not isinstance(type(w_init), np.ndarray):
        raise TypeError(f'TypeError: argument w_init must be <{np.ndarray}>, not <{type(w_init)}>')
    if not isinstance(type(y), float):
        raise TypeError(f'TypeError: argument y must be <{float}>, not <{type(y)}>')
    if not (eta >= 0.) | (eta <= 1.):
        raise ValueError('ValueError: eta must fall between 0. and 1.')
    if thr < 0.:
        raise ValueError('ValueError: thr must be positive')
    if N_max < 0:
        raise ValueError('ValueError: N_max must be positive integer')
    if not x.shape[0] == y.shape[0]:
        raise ValueError('ValueError: x.shape[0] must equal y.shape[0]
    if not x.shape[1] == w_init.shape[0]:
        raise ValueError('ValueError: x.shape[1] must equal w_init.shape[0]
    w_old = w_init
    i = floor(N_max)
    while i > 0:
        #create loop
        if Error(x, w_old, y) < thr:
            return w_old
        w_new = w_old - eta*D_Error(x, w_old, y)
        w_old = w_new
        i -= 1
    return w_old
    """"
