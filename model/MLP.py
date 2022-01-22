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


    def scale_input(self, data):
        if self.preproc["input"] == None:
            return data
        key, a, b = self.preproc["input"]
        if(key == "stand"):
            means, devs = a, b
            return (data-means)/devs
        elif(key == "norm"):
            mins, maxs = a, b
            return (data-mins)/(maxs-mins)
        else:
            raise NotImplementedError("unsupported key")

    def scale_output(self, data):
        if self.preproc["output"] == None:
            return data
        key, a, b = self.preproc["output"]
        if(key == "stand"):
            means, devs = a, b
            return (data-means)/devs
        elif(key == "norm"):
            mins, maxs = a, b
            return (data-mins)/(maxs-mins)
        else:
            raise NotImplementedError("unsupported key")
    
    def unscale_output(self, data):
        if self.preproc["output"] == None:
            return data
        key, a, b = self.preproc["output"]
        if(key == "stand"):
            means, devs = a, b
            return data*devs + means
        elif(key == "norm"):
            mins, maxs = a, b
            return data*(maxs-mins) + mins
        else:
            raise NotImplementedError("unsupported key")

    def scale_dataset(self, dataset):
        return [
            (self.scale_input(pattern[0]), 
             self.scale_output(pattern[1]))
            for pattern in dataset
        ]

    def forward(self, input, training=True):
        for layer in self.layer_set:
            output = layer.forward(input, training)
            input = output
        return output

    def h(self, input):
        out =  self.forward(input, training=False)
        return self.unscale_output(out)
    
    def classify(self, input):
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
