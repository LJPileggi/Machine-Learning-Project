import numpy as np
import h5py

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
        self.activation = activation
        print(f"{layer_dim}; {activation}")
        self.unit_set = [nn_unit(activation, prec_dim) for x in range(layer_dim)]
        self.dropout = dropout if dropout is not None else np.ones((layer_dim,))

    def update_unit_weights(self, eta, lam):
        for unit in self.unit_set:
            unit.update_weights(eta, lam)

    def get_output_dim (self):
        return len(self.unit_set)

    def backwards(self, error_signal):
        cumulative_es = 0 # this gets broadcasted to a all 0 vector
        for i, unit in enumerate(self.unit_set):
#            print(f"\tBackwards dall'unità {i}")
            es_i = unit.backwards(error_signal[i])
#            print(f"np: {new_patterns} : {patt_i}")
            cumulative_es += es_i 
#        print(f"{new_es.ndim}")
        return cumulative_es 
        
    def forward (self, inputs):
        out_list = np.array([])
        for unit in self.unit_set: #magari usare un ufunc
#            print(f"\tRecuperando informazioni dall'unità {unit}\n\t{inputs}")
            out_list = np.append(out_list, unit.forward(inputs))
        return out_list

    def get_weights(self):
        layer_w = []
        for unit in self.unit_set:
            layer_w.append(unit.get_weights())
        return np.array(layer_w)

    def load_weights(self, layer_weights):
        for unit, weights in zip(self.unit_set, layer_weights):
            unit.load_weights(weights)

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
        i = 0
        prec_dim = input_dim
        for layer_dim, activation in zip(layer_struct, activation_set):
            self.layer_set.append(layer(prec_dim, layer_dim, activation=activation))
            prec_dim = layer_dim
            i += 1

    def update_all_weights(self, eta, lam):
        for layer in self.layer_set:
            layer.update_unit_weights(eta, lam)

    def backwards(self, error_signal):
        for layer in reversed(self.layer_set):
#            print(f"\t\tBackwards sul layer {layer}")
            error_signal = layer.backwards(error_signal)
    
    def forward(self, inputs):
        layer_inputs = inputs
        for layer in self.layer_set:
#            print(f"\t\tRecuperando info dal livello {layer}")
            layer_output = layer.forward(layer_inputs)
            layer_inputs = layer_output
        return layer_output

    def save_model (self, filename):
        hf = h5py.File(filename, "w")
        for i, layer in enumerate(self.layer_set):
            layer_weights = layer.get_weights()
            hf.create_dataset(f"layer_{i}", data=layer_weights)
        hf.close()

    def load_model (self, filename):
        hf = h5py.File(filename, "r")
        for i, layer in enumerate(self.layer_set):
            layer_weights = np.array(hf.get(f"layer_{i}"))
            layer.load_weights(layer_weights)
        hf.close()
