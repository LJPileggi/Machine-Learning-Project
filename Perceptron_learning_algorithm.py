#pylint: disable = C0114, C0103

import numpy as np

from activation_functions import threshold_func

def Perceptron_learn_alg(x, w_init, d, eta=0.5, thr=0.01):
    """
    Implements the Perceptron learning algorithm to minimise
    misclassifications.

    params:
     - x: input vector; class type: numpy.ndarray;
     - w_init: initialised weight vectors; class type: numpy.ndarray;
     - d: class classificator, must be either 1 or -1; class type: int;
     - eta: learning rate, must be between 0 and 1; default value: 0.5;
     class type: float;
     - thr: learning threshold, if Dw < thr then returns; must be
     positive; default value: 0.01; class type: float.

     returns:
      - numpy.ndarray of optimal weight values.
    """
    #error handling
    if not isinstance(type(x), np.ndarray):
        raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
    if not isinstance(type(w_init), np.ndarray):
        raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(w_init)}>')
    if not (d == 1) | (d == -1):
        raise ValueError('ValueError: d must be either 1 or -1')
    if not (eta >= 0.) | (eta <= 1.):
        raise ValueError('ValueError: value of eta must fall between 0. and 1.')
    if thr < 0.:
        raise ValueError('ValueError: value of thr must not be less than 0.')
    #actual algorithm
    w_old = w_init
    while True:
        if d == threshold_func(x, w_old, boolean=False):
            return w_old
        w_new = w_old + eta*d*x
        d_w = w_new - w_old
        if np.all((d_w < thr) & (d_w > -thr)):
            return w_new
