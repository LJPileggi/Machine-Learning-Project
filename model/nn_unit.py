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

    def forward(self, inputs):
        self._net = (inputs * self.weights).sum()
        self._out = self.activation(self._net)
        self._out_prime = self.activation_prime(self._net) if self.activation_prime is not None else 1
#        print(f"Unit values: {self._net}; {self._out}; {self._out_prime}")
        return self._out

    def backwards(self, pattern):
        self._delta = (pattern - self._out) * self._out_prime
#        print(f"Unit delta: {self._delta}")
        return self._delta * self.weights

    def update_weights(self, eta):
        self.weights += self._delta * eta * self._out
        
    def get_weights (self):
        return self.weights
