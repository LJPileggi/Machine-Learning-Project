import numpy as np

from nn_unit import nn_unit
import activation_functions

class layer:
    """
    Single layer of the MLP.

    Attributes:
     - layer_dim: number of units of the layer; class type: int;
     - layer_prec: key to the previous layer; class type: layer;
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
    def __init__(self, prec_dim, layer_dim, activation = "linear", dropout=None):
        self.layer_prec = layer_prec #None == input layer
        self.activation = activation
        print(f"{layer_dim}; {activation}")
        self.unit_set = [nn_unit(activation, prec_dim) for x in range(layer_dim)]
        self.dropout = dropout if dropout is not None else np.ones((layer_dim,))

    def update_unit_weights(self, eta):
        for i in range(len(self.unit_set)):
#            print(f"{i}: {weights_coef[i]}")
            unit = self.unit_set[i]
            unit.update_weights(eta)

    def get_output_dim (self):
        return len(self.unit_set)

    def backwards(self, pattern):
        new_patterns = None
        for i in range(len(self.unit_set)):
            unit = self.unit_set[i]
#            print(f"\tBackwards dall'unità {i}")
            patt_i = unit.backwards(pattern[i])
#            print(f"np: {new_patterns} : {patt_i}")
            new_patterns = patt_i if new_patterns is None else np.append(new_patterns, patt_i, axis=0) 
#        print(f"{new_patterns.ndim}")
        return new_patterns if new_patterns.ndim == 1 else np.sum(new_patterns, axis=1) #da controllare se gli assi son giusti. looks like it!
        
    def forward (self, inputs):
        out_list = []
        for i in range(len(self.unit_set)): #magari usare un ufunc
            unit = self.unit_set[i]
#            print(f"\tRecuperando informazioni dall'unità {i}\n\t{inputs}")
            out_list.append(unit.forward(inputs))
        return out_list
        

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
    def __init__(self, input_dim, layer_struct, activation_set):
        self.layer_struct = layer_struct
        self.layer_set = []
        layer_prec = None
        i = 0
        prec_dim = input_dim
        for layer_dim, activation in zip(layer_struct, activation_set):
            self.layer_set.append(layer(prec_dim, layer_dim, activation=activation, layer_prec=layer_prec))
            prec_dim = layer_dim
            i += 1

    def update_all_weights(self, eta):
        for i in range(len(self.layer_set)):
            layer = self.layer_set[i]
            layer.update_unit_weights(eta)

    def backwards(self, pattern):
        for i in range(len(self.layer_set)):
            layer = self.layer_set[-i-1]
#            print(f"\t\tBackwards sul layer {-i-1}")
            pattern = layer.backwards(pattern)
    
    def forward(self, inputs):
        layer_inputs = inputs
        for i in range(len(self.layer_set)):
            layer = self.layer_set[i]
#            print(f"\t\tRecuperando info dal livello {i}")
            layer_output = layer.forward(layer_inputs)
            layer_inputs = layer_output
        return layer_output
