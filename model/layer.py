from os import error
import numpy as np
import activation_functions

class layer:
    """
    Single layer of the MLP.

    Constructor arguments:
     - input_dim: size of the input vector/previous layer output
     - layer_dim: number of units of the layer; class type: int;
     - activation: activation function of the various units; class type: str
     - dropout: tells which units are active or deactivated;
       default value: numpy.ones(layer_dim); class type: numpy.ndarray. 

    Attributes:
     - _WM: Weight matrix, n rows and m columns, where eeach column is the weight vector of 1 unit.
     - _input: array of input; class type: numpy.ndarray;
     - _output_prime: array of output_prime values from units; class type: numpy.ndarray;
     - _activation:
     - _activation_prime:
     - _dropout: dropout mask

    Methods:
     - forward
     - backward
     - update_weights
    """
    def __init__(self, input_dim, layer_dim, activation = "sigmoidal", dropout=None):
        print(f"{layer_dim}; {activation}")
        self._WM = np.random.normal(loc=0.0, scale=0.5, size=(input_dim, layer_dim) )
        self._biases = np.random.normal(loc=0.0, scale=0.5, size=layer_dim )
        self._negGrad = 0
        self._biases_negGrad = 0
        self._DWold = 0
        self._biases_DWold = 0
        self.inputs = None
        self.output_prime = None
        if activation != "sigmoidal":
          raise Exception("activation function Not implemented yet")
        self._activation = activation_functions.sigmoidal
        self._act_prime = activation_functions.d_sigmoidal
        self._dropout = dropout if dropout is not None else np.ones((layer_dim,))

    def forward (self, inputs):
      net = np.dot(inputs, self._WM) + self._biases    #computes the net of all units at one
      output = self._activation(net)     #computes the out of all units at one

      self.inputs = inputs                      #stores inputs, for backprop calculation
      self.output_prime = self._act_prime(net)  #stores out_pr, for backprop calculation
      return output
    
    def backwards(self, error_signal):
        #deltas is the vector (d_t1, d_t2, d_t3..), for each unit t1, t2, t3 ..
        deltas = error_signal * self.output_prime 
        
        #here we compute the negative gradient for the current layer, but we don't apply it yet
        #   we add to the total neg gradient the current contribution, ie the neg grad calculated from this pattern
        #   the outer product produes the matrix array([ d_t*input for each unit t in layer])
        self._negGrad += np.outer(self.inputs, deltas) 
        self._biases_negGrad += deltas #for each unit t, Dbias_t = d_t * input, where input is 1

        #here we compute the error signal for the previous layer
        #    deltas, seen as a row vector, is broadcasted to a matrix with all equal rows.
        #    The point-wise multiplicaion multiplies the w vector of unit t with it's corresponding d_t
        #    (the w vector of unit t is just a column vetor in the WM matrix).
        #    By summing on the orizontal axis, we get a column vector: the error signal for each input,
        #    i.e. the error signal for each unit in the previous layer
        return np.sum(self._WM*deltas, axis=1)


    def update_weights(self, eta, lam, alpha):
        #here we apply the negGradient computed during backward.
        #In a online alg, we continously call this.backward and this.update_weights,
        #   so for each pattern, negGrad is computed and applied immediately.
        #In a (mini)batch alg, we call this.backward for each pattern, and
        #   this.update_weights just once per batch.
        DW = eta*self._negGrad + alpha*self._DWold
        self._WM += DW - lam*self._WM
        self._DWold = DW
        
        Dbias = eta * self._biases_negGrad + alpha*self._biases_DWold
        self._biases += Dbias
        self._biases_DWold = Dbias

        self._negGrad = 0
        self._biases_negGrad = 0
    
    def get_weights(self):
      return (self._WM, self._biases)

    def load_weights(self, weights, biases):
        self._WM = weights
        self._biases = biases
        