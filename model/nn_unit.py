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
    def __init__(self, activation, dim):
        self.weights = np.random.randn(dim)
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

    def _net(self, inputs):
        """
        returns argument for activation function. Type: float
        """
        self._net_value = (inputs * self.weights).sum()
        return self._net_value

    def out(self, inputs):
        """
        returns output units. Type: either float or numpy.ndarray
        """
        self._out_value = self.activation(self._net(inputs))
        return self._out_value

    def out_prime(self, inputs):
        """
        returns output of derivative of activation function.
        Type: either float or numpy.ndarray
        """
        self._out_prime_value = self.activation_prime(self._net(inputs)) if self.activation_prime is not None else 1
        return self._out_prime_value

    def update_weights(self, weights_new):
        """
        Updates weights of unit.
        
        Wouldn't it be more convenient though to use properties and setters?
        """
        self.weights = weights_new

    def get_weights (self):
        return self.weights

    def get_outputs (self):
        return self._out_value, self._out_prime_value
