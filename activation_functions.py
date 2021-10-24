#pylint: disable = C0114, C0103, C0301, R1716

import numpy as np

def lin_func(x, w_i):
    """
    Linear activation function for NN unit output.

    params:
     - x: input vector; class type: numpy.ndarray;
     - w_i: weight vector; class type: numpy.ndarray.

    returns:
     - scalar product between x and w_i; class type: float.
    """
    if not isinstance(type(x), np.ndarray):
        raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
    if not isinstance(type(w_i), np.ndarray):
        raise TypeError(f'TypeError: argument w_i must be <{np.ndarray}>, not <{type(w_i)}>')
    return float((x * w_i).sum())

def threshold_func(x, w_i, boolean=True):
    """
    Threshold activation function for NN unit output.

    params:
     - x: input vector; class type: numpy.ndarray;
     - w_i: weight vector; class type: numpy.ndarray;
     - boolean: sets the output to 1/0 (True) or 1/-1 (False);
     set by default to True; class type: bool.

    returns:
     - 1./0. (boolean=True); 1./-1. (boolean=False).
    """
    if not isinstance(type(x), np.ndarray):
        raise TypeError(f'TypeError: argument x must be <{np.ndarray}>, not <{type(x)}>')
    if not isinstance(type(w_i), np.ndarray):
        raise TypeError(f'TypeError: argument w_i must be <{np.ndarray}>, not <{type(w_i)}>')
    if boolean:
        if float((x * w_i).sum()) >= 0.:
            return 1.
        return 0.
    if float((x * w_i).sum()) >= 0.:
        return 1.
    return -1.

def sigmoidal(x, w_i, a=1., thr=0., hyperbol=False):
    """
    Sigmoidal activation function for NN unit output.
    Smooth and differentiable approximation to the threshold function.

    params:
     - x: input vector; class type: numpy.ndarray;
     - w_i: weight vector; class type: numpy.ndarray;
     - a: exponent parameter; set by default to 1.; class type: float;
     - thr: sets the rejection zone of the model;
     must be between 0 and 1; set by default to 0.; class type: float;
     - hyperbol: sets the interval of output values either to [0., 1.] (False)
       or to [-1., 1.] (True); set by default to False; class type: bool.

    returns:
     - 1./0. (hyperbol=False); 1./-1. (hyperbol=True).
    """
    if (thr > 1.) | (thr < 0.):
        raise ValueError('ValueError: invalid value for argument thr. Accepted values between 0. and 1. only')
    z = lin_func(x, w_i)
    if not hyperbol:
        out = 1./(1. + np.exp(-a*z))
        if thr != 0.:
            if ((out > 0.5*(1. - thr)) and (out < 0.5*(1. + thr))):
                raise ValueError('ValueError: unit output falls within rejection zone')
        if out >= 0.5*(1. + thr):
            return 1.
        return 0.
    out = np.tanh(-a*z/2.)
    if thr!= 0.:
        if ((out > (1. - thr)) and (out < (1. + thr))):
            raise ValueError('ValueError: unit output falls within rejection zone')
    if out >= 1. + thr:
        return 1.
    return -1.
