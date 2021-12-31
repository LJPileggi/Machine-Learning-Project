import numpy as np
import random
import h5py

from layer import layer

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

class MLP:
    """
    Multi-layer Perceptron.

    Attributes:
     - layer_struct: list of dimensions of single layers; class type: list;
     - inputs: inputs of the whole network; class type: numpy.ndarray;
     - activation_set: list of activation functions of various layers; class type: list of functions.


    Methods:
     - forawrd:
     . backwards: 
     - update_weights: update weights of the whole network; returns None
    """
    def __init__(self, input_dim, architecture, seed=4444):
        set_seed(seed)
        self.layer_set = []
        prec_dim = input_dim
        for layer_dim, activation in architecture:
            self.layer_set.append(layer(prec_dim, layer_dim, activation))
            prec_dim = layer_dim


    def forward(self, input):
        for layer in self.layer_set:
            output = layer.forward(input)
            input = output
        return output

    def backwards(self, error_signal):
        for layer in reversed(self.layer_set):
            error_signal = layer.backwards(error_signal)
    
    def update_weights(self, eta, lam, alpha):
        for layer in self.layer_set:
            layer.update_weights(eta, lam, alpha)
    
    def get_max_grad_list(self):
        return [layer.get_max_grad() for layer in self.layer_set]


    def save_model (self, filename):
        hf = h5py.File(filename, "w")
        for i, layer in enumerate(self.layer_set):
            layer_weights, biases = layer.get_weights()
            hf.create_dataset(f"layer_{i}_weights", data=layer_weights)
            hf.create_dataset(f"layer_{i}_biases", data=biases)
        hf.close()

    def load_model (self, filename):
        hf = h5py.File(filename, "r")
        for i, layer in enumerate(self.layer_set):
            layer_weights = np.array(hf.get(f"layer_{i}_weights"))
            baiases = np.array(hf.get(f"layer_{i}_biases"))
            layer.load_weights(layer_weights, baiases)
        hf.close()
            
