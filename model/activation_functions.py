#pylint: disable = C0114, C0103, C0301, R1716

import numpy as np

#Linear activation function
def linear(network_value):
    """
    Linear activation function for NN unit output.

    params:
     - network_value: the scalar obtained once calculated value of the network with their current weights. class type = float;

    returns:
     - The identity; class type: float.
    """
    return network_value

def d_linear(network_value):
    """
    Derivative of linear activation function.

    params:
     - network_value: the scalar obtained once calculated value of the network with their current weights. class type = float;

    returns:
     - 1.
    """
    return 1.

def threshold(network_value, boolean=True):
    """
    Threshold activation function for NN unit output.

    params:
     - network_value: the scalar obtained once calculated value of the network with their current weights. class type: float;

    returns:
     - 1/0 (boolean=True); 1/-1 (boolean=False).
    """
    if boolean:
        return network_value >= 0.
    else:
        return 1 if network_value >= 0. else -1

def tanh(network_value, thr=0.):
    return np.tanh(network_value)
    # out =  np.tanh(network_value)
    # if thr == 0.: 
    #     return out
    # else:
    #     if (thr > 1.) | (thr < 0.):
    #         raise ValueError('ValueError: invalid value for argument thr. Accepted values between 0. and 1. only')
    #     if ((out > -thr) and (out <  thr)):
    #         raise ValueError('ValueError: unit output falls within rejection zone') #mmm
    #     else:
    #         return 1 if out >= 0. else -1


def d_tanh(network_value):
    return 1-(tanh(network_value)**2)

def sigmoidal(network_value, a=1., thr=0.):
    """
    Sigmoidal activation function for NN unit output.
    Smooth and differentiable approximation to the threshold function.

    params:
     - network_value: the scalar obtained once calculated value of the network with their current weights. class type = float
     - a: exponent parameter; set by default to 1.; class type: float;
     - thr: sets the rejection zone of the model;
     must be between 0 and 1; set by default to 0.; class type: float;
     - hyperbol: sets the interval of output values either to [0., 1.] (False)
       or to [-1., 1.] (True); set by default to False; class type: bool.

    returns:
     - 1/0 (hyperbol=False); 1/-1 (hyperbol=True).
    """
    return 1./(1. + np.exp(-a*network_value))
    # out = 1./(1. + np.exp(-a*network_value))
    # if thr == 0.: 
    #     return out
    # else:
    #     if (thr > 1.) | (thr < 0.):
    #         raise ValueError('ValueError: invalid value for argument thr. Accepted values between 0. and 1. only')
    #     if ((out > 0.5*(1. - thr)) and (out < 0.5*(1. + thr))): 
    #         raise ValueError('ValueError: unit output falls within rejection zone') #perché ritornare un errore?
    #     else:
    #         return 1 if out >= 0.5*(1. + thr) else 0




def d_sigmoidal(network_value, a=1.):
    """
    Derivative of sigmoidal function.

    params:
     - network_value: the scalar obtained once calculated value of the network with their current weights. class type = float
     - a: exponent parameter; set by default to 1.; class type: float;
     - hyperbol: determines whether the derived function is a hyperbolic tangent
     (True) or sigmoid (False). Default value: False.
    """
    k = np.exp(a*network_value)
    out = k / ((k+1)**2)
    return out

def ReLu(network_value):
    """
    ReLu activation function. Its derivative is simply the boolean threshold function.
    Returns the identity if network_value > 0, 0 otherwise.
    """
    return np.maximum(0, network_value)


def d_ReLU(network_value):
    return np.where(network_value > 0., 1., 0.)


def softplus(network_value, a=1.):
    """
    Softplus activation function. Gives a smooth approximation of ReLu function.
    """
    out = np.log(1 + np.exp(a*network_value))/a
    return out
# usoftplus = np.frompyfunc(softplus, 1, 1)

def d_softplus(network_value, a=1.):
    """
    Derivative of softplus.
    """
    out = 1./(1. + np.exp(-a*network_value))
    return out

def gaussian(network_value, a=0.5):
    """
    Gaussian activation function.
    """
    out = np.exp(-a*network_value**2)
    return out

def d_gaussian(network_value, a=0.5):
    """
    Derivative of gaussian.
    """
    out = -2.*a*network_value*np.exp(-a*network_value**2)
    return out
def SiLu(network_value, a=1.):
    """
    Sigmoidal linear unit: other smooth approximation for ReLu.
    """
    out = network_value/(1. + np.exp(-a*network_value))
    return out

def d_SiLu(network_value, a=1.):
    """
    Derivative of SiLu.
    """
    out = np.exp(a*network_value)*(a*network_value + np.exp(a*network_value) + 1.)/(np.exp(a*network_value) + 1.)**2
    return out

usoftplus = np.frompyfunc(lambda x: softplus(x, 0.5), 1, 1)

