import numpy as np
import random
import h5py

import layer

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
    def __init__(self, seed, task, input_dim, architecture, preproc):
        set_seed(seed)
        self.preproc = preproc
        self.layer_set = []
        prec_dim = input_dim
        for activation, options in architecture:
            if activation == "sigmoidal":
                self.layer_set.append(layer.Sigmoidal(prec_dim, options))
                prec_dim = options
            elif activation == "tanh":
                self.layer_set.append(layer.Tanh(prec_dim, options))
                prec_dim = options
            elif activation == "linear":
                self.layer_set.append(layer.Linear(prec_dim, options))
                prec_dim = options
            elif activation == "relu":
                self.layer_set.append(layer.ReLu(prec_dim, options))
                prec_dim = options
            elif activation == "silu":
                self.layer_set.append(layer.SiLu(prec_dim, options))
                prec_dim = options
            elif activation == "softplus":
                self.layer_set.append(layer.Softplus(prec_dim, options))
                prec_dim = options
            elif activation == "BatchNormalization":
                self.layer_set.append(layer.BatchNormalization(options))
            elif activation == "dropout":
                self.layer_set.append(layer.Dropout(options))
        last_layer_fun = architecture[-1][0]
        self.threshold = None
        if last_layer_fun == "sigmoidal" and task == "classification":
            self.threshold = 0.5
        elif last_layer_fun == "tanh" and task == "classification":
            self.threshold = 0.
        elif last_layer_fun == "linear" and task == "regression":
            self.threshold == None
        else:
            raise NotImplementedError("unsupported activation function in output layer")

    def scale(self, data, component):
        shift, scale = self.preproc[component]
        return (data-shift)/scale
    
    def unscale(self, data, component):
        shift, scale = self.preproc[component]
        return data*scale + shift

    def scale_dataset(self, dataset):
        return [
            (self.scale(pattern[0], "input"), 
             self.scale(pattern[1], "output"))
            for pattern in dataset
        ]

    def forward(self, input, training=True):
        for layer in self.layer_set:
            output = layer.forward(input, training)
            input = output
        return output

    def predict(self, input):
        input = self.scale(input, "input")
        output =  self.forward(input, training=False)
        output = self.unscale(output, "output")
        return output
    
    def classify(self, input):
        input = self.scale(input, "input")
        out =  self.forward(input, training=False)
        return out > self.threshold

    def backwards(self, error_signal):
        for layer in reversed(self.layer_set):
            error_signal = layer.backwards(error_signal)
    
    def update_weights(self, eta, lam, alpha):
        for layer in self.layer_set:
            layer.update_weights(eta, lam, alpha)
    
    def get_max_grad_list(self):
        return [layer.get_max_grad() for layer in self.layer_set]

    def get_weights(self):
        return np.concatenate([layer.get_weights() for layer in self.layer_set])
