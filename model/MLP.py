import numpy as np

from nn_unit import nn_unit
import activation_functions

class layer:
    """
    Single layer of the MLP.

    Attributes:
     - layer_dim: number of units of the layer; class type: int;
     - layer_prec: key to the previous layer; class type: int;
     - inputs; inputs of the layer; class type: numpy.ndarray;
     - weights: matrix of weights of the single units; 
       class type: numpy.ndarray of shape(#outputs, #inputs);
     - activation: activation function of the various units; class type: str
     - dropout: tells which units are active or deactivated;
       default value: numpy.ones(layer_dim); class type: numpy.ndarray. 

    Private attributes:
     - _net_set: array of net values from units; class type: numpy.ndarray;
     - _output_set: array of output values from units; class type: numpy.ndarray;
     - _output_prime_set: array of output_prime values from units; class type: numpy.ndarray;

    Methods:
     - initialise_weight_matrix: uploads units' weights into the weight matrix;
       returns None;
     - update_unit_inputs: for each unit, calls the update_inputs method;
       returns None;
     - update_unit_weights: update each unit's weights with the ones in the
       weight matrix; returns None
    """
    def __init__(self, layer_dim, activation = "linear", layer_prec = None, dropout=None):
        self.layer_prec = layer_prec #None == input layer
        prec_dim = self.layer_prec.get_output_dim() if self.layer_prec is not None else 6 #per ora hardcoded, poi ci pensiamo
        self.activation = activation
        print(f"{layer_dim}; {activation}")
        self.unit_set = {x: nn_unit(activation, prec_dim) for x in range(layer_dim)}
        self.initialize_weight_matrix()
        #self._net_set = np.array([self.unit_set[k]._net(inputs) for k in self.unit_set])
        #self._output_set = np.array([self.unit_set[k].out(inputs) for k in self.unit_set])
        #self._output_prime_set = np.array([self.unit_set[k].out_prime(inputs) for k in self.unit_set])
        self.dropout = dropout if dropout is not None else np.ones((layer_dim,))

    def initialize_weight_matrix(self):
        print(self.unit_set[0])
        self.weights = np.array([self.unit_set[k].get_weights() for k in range(len(self.unit_set))])

    def update_unit_weights(self):
        for i, unit in self.unit_set:
            unit.update_weights(self.weights[i])

    def get_output_dim (self):
        return len(self.unit_set)

class MLP:
    """
    Multi-layer Perceptron.

    Attributes:
     - layer_struct: list of dimensions of single layers; class type: list;
     - inputs: inputs of the whole network; class type: numpy.ndarray;
     - activation_set: list of activation functions of various layers; class type: list of functions.

    Private attributes:
     - _output: returns output of the whole network; class type: numpy.ndarray.

    Methods:
     - update_network_inputs: update inputs of the whole network; returns None;
     - update_all_weights: update weights of the whole network; returns None
    """
    def __init__(self, layer_struct, activation_set):
        self.layer_struct = layer_struct
        self.inputs = None
        self.layer_set = {}
        layer_inputs = self.inputs
        layer_prec = None
        i = 0
        for layer_dim, activation in zip(layer_struct, activation_set):
            self.layer_set.update({i : layer(layer_dim, activation=activation, layer_prec=layer_prec)})
            layer_prec = self.layer_set[i]
            i += 1

    def update_all_weights(self):
        for layer in NN.layer_set:
            layer.update_unit_weights()
