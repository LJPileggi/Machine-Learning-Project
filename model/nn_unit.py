#pylint: disable = C0114, C0103, R0913

import numpy as np
import enum

import activation_functions

class nn_unit:
    """
    Object class for the NN unit
    Attributes:
     - inputs: inputs of the unit; class type: numpy.ndarray;
     - activation: list containing, in order, the name of the activation func (str) and its
     params in the order they appear in such func.

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
        if activation[0] == "linear":
            self.activation = linear
            self.activation_prime = d_linear
        elif activation[0] == "threshold":
            if activation[1] == True:
                self.activation = threshold
                self.activation_prime = None
            else:
                self.activation = threshold(boolean=False)
                self.activation_prime = None
        elif activation[0] == "sigmoidal":
            if activation[3] == False:
                self.activation = sigmoidal(a=activation[1], thr=activation[2])
                self.activation_prime = sigmoidal(a=activation[1])
            else:
                self.activation = sigmoidal(a=activation[1], thr=activation[2], hyperbol=True)
                self.activation_prime = sigmoidal(a=activation[1], hyperbol=True)
        elif activation[0] == "softplus":
            self.activation = softplus(a=activation[1])
            self.activation_prime = d_softplus(a=activation[1])
        elif activation[0] == "gaussian":
            self.activation = gaussian(a=activation[1])
            self.activation_prime = d_gaussian(a=activation[1])
        elif activation[0] == "SiLu":
            self.activation = SiLu(a=activation[1])
            self.activation_prime = d_SiLu(a=activation[1])
        elif activation[0] == "ReLu":
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
