#pylint: disable = C0114, C0103

from math import floor

import numpy as np

from activation_functions import threshold_func

def Perceptron_learn_alg(x, w_init, d, eta=0.5, thr=0.01, N_max=1000):
    """
    Implements the Perceptron learning algorithm to minimise
    misclassifications. Starts from an initialised weight
    vector: if no misclassification is present, returns
    such vector itself; otherwise, updates weight vector
    with a correction term, until changes go below a
    specified threshold.

    params:
     - x: input vector; class type: numpy.ndarray;
     - w_init: initialised weight vectors; class type: numpy.ndarray;
     - d: class classificator, must be either 1 or -1; class type: int;
     - eta: learning rate, must lie between 0 and 1; default value: 0.5;
     class type: float;
     - thr: learning threshold, if Dw < thr then returns; must be
     positive; default value: 0.01; class type: float;
     - N_max: control parameter, sets the maximum number of iterations
     handlable; must be positive integer; default value: 1000;
     class type: int.

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
        raise ValueError('ValueError: eta must fall between 0. and 1.')
    if thr < 0.:
        raise ValueError('ValueError: thr must be positive')
    if N_max < 0:
        raise ValueError('ValueError: N_max must be positive integer')
    #actual algorithm
    w_old = w_init
    i = floor(N_max)
    while i > 0:
        if d == threshold_func(x, w_old, boolean=False):
            return w_old
        w_new = w_old + eta*d*x
        d_w = w_new - w_old
        if np.all((d_w < thr) & (d_w > -thr)):
            return w_new
        w_old = w_new
        i -= 1
    return w_old
