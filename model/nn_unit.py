#pylint: disable = C0114, C0103, R0913

import numpy as np
import enum

from activation_functions import *

class nn_unit:
    """
    Object class for the NN unit
    Attributes:
     - inputs: inputs of the unit; class type: numpy.ndarray;
     - activation: string containing the name of the activation func (str) 

    Private attributes:
     - net: scalar product between x and w_i; class type: float.

    Methods:
     - out: returns output vector; returns float or numpy.ndarray.
     - out_prime: returns derivative of output vector; returns float or numpy.ndarray;
     - update_inputs: updates inputs of unit; returns None;
     - update_weights: updates weights of unit; returns None.
    """
    def __init__(self, inputs, activation):
        self.inputs = inputs
        self.weights = np.random.randn(len(inputs))
        if activation == "linear":
            self.activation = linear
            self.activation_prime = d_linear
        elif activation == "threshold":
            self.activation = threshold
            self.activation_prime = None
        elif activation == "tanh":
            self.activation = tanh
            self.activation_prime = d_tanh
        elif activation == "sigmoidal":
            self.activation = sigmoidal
            self.activation_prime = d_sigmoidal
        elif activation == "softplus":
            self.activation = softplus
            self.activation_prime = d_softplus
        elif activation == "gaussian":
            self.activation = gaussian
            self.activation_prime = d_gaussian
        elif activation == "SiLu":
            self.activation = SiLu
            self.activation_prime = d_SiLu
        elif activation == "ReLu":
            self.activation = ReLu
            self.activation_prime = threshold

    def _net(self):
        """
        returns argument for activation function. Type: float
        """
        return (self.inputs * self.weights).sum()

    def out(self):
        """
        returns output units. Type: either float or numpy.ndarray
        """
        return self.activation(self._net(self.inputs))

    def out_prime(self):
        """
        returns output of derivative of activation function.
        Type: either float or numpy.ndarray
        """
        return self.activation_prime(self._net(self.inputs))

    def update_inputs(self, inputs_new):
        """
        Updates inputs of unit.
        
        Wouldn't it be more convenient though to use properties and setters?
        """
        self.inputs = inputs_new

    def update_weights(self, weights_new):
        """
        Updates weights of unit.
        
        Wouldn't it be more convenient though to use properties and setters?
        """
        self.weights = weights_new
