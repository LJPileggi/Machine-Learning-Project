from os import error
import numpy as np
import math
import activation_functions

class Layer():
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
    def __init__(self):
        self._negGrad = 0.
        self._last_max_grad = 0.
        self._biases_negGrad = 0.
        self._DWold = 0.
        self._biases_DWold = 0.
        self.inputs = None
        self.output_prime = None
        self._WM = None
        self._biases = None

    def activation (self, network_value):
        raise NotImplementedError ("This layer doesn't have an activation Function")

    def activation_prime (self, network_value):
        raise NotImplementedError ("This layer doesn't have an activation Function")
      
    def forward_old (self, inputs, training=True):
        self.inputs = inputs              #stores inputs, for backprop calculation
        net = np.dot(self.inputs, self._WM) + self._biases    #computes the net of all units at one
        output = self.activation(net)     #computes the out of all units at one
        if training: #tbh non so questo
            self.output_prime = self.activation_prime(net)  #stores out_pr, for backprop calculation
        return output

    def forward (self, inputs, training=True):
        self.inputs = inputs
        if (inputs.ndim == 1):
            net = np.dot(self.inputs, self._WM) + self._biases    #computes the net of all units at one
            output = self.activation(net)     #computes the out of all units at one
            if training: #tbh non so questo
                self.output_prime = self.activation_prime(net)  #stores out_pr, for backprop calculation
        else:
            output = []
            self.output_prime = []
            for inp in self.inputs:
                #print(f"{inp.shape}")
                net = np.dot(inp, self._WM) + self._biases
                #print(f"{net}")
                output.append(self.activation(net))
                if training:
                    self.output_prime.append(self.activation_prime(net))
            self.output_prime=np.array(self.output_prime)
            output=np.array(output)
        return output
    
    def backwards_old(self, error_signal):
        #deltas is the vector (d_t1, d_t2, d_t3..), for each unit t1, t2, t3 ..
        deltas = error_signal * self.output_prime
        
        #here we compute the negative gradient for the current layer, but we don't apply it yet
        #   we add to the total neg gradient the current contribution, ie the neg grad calculated from this pattern
        #   the outer product produes the matrix array([ d_t*input for each unit t in layer])
        #print(f"{self.inputs.shape} - {deltas.shape}")
        self._negGrad += np.outer(self.inputs, deltas)
        self._biases_negGrad += deltas #for each unit t, Dbias_t = d_t * input, where input is 1

        #here we compute the error signal for the previous layer
        #    deltas, seen as a row vector, is broadcasted to a matrix with all equal rows.
        #    The point-wise multiplicaion multiplies the w vector of unit t with it's corresponding d_t
        #    (the w vector of unit t is just a column vetor in the WM matrix).
        #    By summing on the orizontal axis, we get a column vector: the error signal for each input,
        #    i.e. the error signal for each unit in the previous layer
        return np.sum(self._WM*deltas, axis=1) #forse dovremmo cambiarlo con deltas @ self._WM

    def backwards(self, error_signal):
        #print(f"{error_signal.shape}")
        deltas = error_signal * self.output_prime
        if(error_signal.ndim == 1):
            #here we compute the negative gradient for the current layer, but we don't apply it yet
            #   we add to the total neg gradient the current contribution, ie the neg grad calculated from this pattern
            #   the outer product produes the matrix array([ d_t*input for each unit t in layer])
            #print(f"{self.inputs.shape} - {deltas.shape}")
            self._negGrad += np.outer(self.inputs, deltas)
            self._biases_negGrad += deltas #for each unit t, Dbias_t = d_t * input, where input is 1

            #here we compute the error signal for the previous layer
            #    deltas, seen as a row vector, is broadcasted to a matrix with all equal rows.
            #    The point-wise multiplicaion multiplies the w vector of unit t with it's corresponding d_t
            #    (the w vector of unit t is just a column vetor in the WM matrix).
            #    By summing on the orizontal axis, we get a column vector: the error signal for each input,
            #    i.e. the error signal for each unit in the previous layer
            return np.sum(self._WM*deltas, axis=1) #forse dovremmo cambiarlo con deltas @ self._WM
        else:
            for i, delta in enumerate(deltas):
                for j, inp in enumerate(self.inputs):
                    if (i == j):
                        self._negGrad += np.outer(inp, delta)
            self._biases_negGrad += np.sum(deltas, axis=0)
            return deltas @ self._WM.T


    def update_weights(self, eta, lam, alpha):
        #here we apply the negGradient computed during backward.
        #In a online alg, we continously call this.backward and this.update_weights,
        #   so for each pattern, negGrad is computed and applied immediately.
        #In a (mini)batch alg, we call this.backward for each pattern, and
        #   this.update_weights just once per batch.
        DW = eta * self._negGrad + alpha*self._DWold
        self._WM += DW - lam*self._WM
        self._DWold = DW
        
        Dbias = eta * self._biases_negGrad + alpha*self._biases_DWold
        self._biases += Dbias
        self._biases_DWold = Dbias


        self._last_max_grad = np.amax(np.absolute(self._negGrad))
        self._negGrad = 0.
        self._biases_negGrad = 0
    
    def get_max_grad(self):
      return self._last_max_grad

    def get_weights(self):
      return np.vstack((self._WM, self._biases)).flatten() #if (self._WM != None and self._biases != None) else [0]
        
class Sigmoidal(Layer):

    def __init__(self, input_dim, layer_dim):
        print(f"{layer_dim}; Sigmoidal")
        super().__init__()
        #lower, upper = -1./(math.sqrt(input_dim)), 1./(math.sqrt(input_dim)) #xavier
        lower, upper = -math.sqrt(6)/(math.sqrt(input_dim + layer_dim)), math.sqrt(6)/(math.sqrt(input_dim + layer_dim)) #normalized xavier
        self._WM = np.random.uniform(low = lower, high = upper, size=(input_dim, layer_dim) )
        self._biases = np.random.uniform(low = lower, high = upper, size=layer_dim )
        #self._WM = np.random.normal(loc=0.0, scale=0.5, size=(input_dim, layer_dim) )
        #self._biases = np.random.normal(loc = 0.0, scale = 0.5, size=layer_dim )

    def activation (self, network_value, a=1.):
        return 1./(1. + np.exp(-a*network_value))

    def activation_prime (self, network_value, a=1.):
        k = np.exp(a*network_value)
        out = k / ((k+1)**2)
        return out

class Tanh(Layer):

    def __init__(self, input_dim, layer_dim):
        print(f"{layer_dim}; Tanh")
        super().__init__()
        #lower, upper = -1./(math.sqrt(input_dim)), 1./(math.sqrt(input_dim)) #xavier
        lower, upper = -math.sqrt(6)/(math.sqrt(input_dim + layer_dim)), math.sqrt(6)/(math.sqrt(input_dim + layer_dim)) #normalized xavier
        self._WM = np.random.uniform(low = lower, high = upper, size=(input_dim, layer_dim) )
        self._biases = np.random.uniform(low = lower, high = upper, size=layer_dim )
        #self._WM = np.random.normal(loc=0.0, scale=0.5, size=(input_dim, layer_dim) )
        #self._biases = np.random.normal(loc = 0.0, scale = 0.5, size=layer_dim )

    def activation (self, network_value):
        return np.tanh(network_value)

    def activation_prime (self, network_value):
        return 1-(np.tanh(network_value)**2)

class Linear(Layer):

    def __init__(self, input_dim, layer_dim):
        print(f"{layer_dim}; Linear")
        super().__init__()
        lower, upper = -math.sqrt(6)/(math.sqrt(input_dim + layer_dim)), math.sqrt(6)/(math.sqrt(input_dim + layer_dim)) #normalized xavier
        self._WM = np.random.uniform(low = lower, high = upper, size=(input_dim, layer_dim) )
        self._biases = np.random.uniform(low = lower, high = upper, size=layer_dim )

    def activation (self, network_value):
        return network_value

    def activation_prime (self, network_value):
        return [1. for ele in network_value]

class ReLu(Layer):

    def __init__(self, input_dim, layer_dim):
        print(f"{layer_dim}; ReLu")
        super().__init__()
        self._WM = np.random.normal(loc=0.0, scale=math.sqrt(2/input_dim), size=(input_dim, layer_dim)) * math.sqrt(2./input_dim) #He weight initialization
        self._biases = np.zeros(shape=(layer_dim,))

    def activation (self, network_value):
        return np.maximum(0, network_value)

    def activation_prime (self, network_value):
        return np.where(network_value > 0, 1., 0.)

class BatchNormalization(Layer):

    def __init__(self, input_dim, layer_dim):
        print(f"{layer_dim}; BatchNormalization")
        super().__init__()
        self._WM = np.random.normal(loc=0.0, scale=0.5, size=(input_dim, layer_dim) ) #in realtà ha pesi diversi
        self._biases = np.random.normal(loc=0.0, scale=0.5, size=layer_dim )
        

class Dropout(Layer): #potremmo benissimo trasformarlo in un layer tutto suo

    def __init__(self, input_dim, rate):
        print(f"Dropout: {rate}")
        super().__init__()
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError (f"Invalid Value {rate} received - `rate` needs to be betweek 0 and 1")
        self.input_dim = input_dim
        #self.activated_inputs = np.random.choice([1., 0.], size=(self.input_dim, ), p=[rate, 1-rate])
        self.scale = 1./(1.-rate)
        self.rate = rate

    def forward(self, inputs, training=True):
        if (training):
            self.activated_inputs = np.random.choice([1., 0.], size=inputs.shape, p=[self.rate, 1-self.rate])
            return (self.scale * inputs) * self.activated_inputs
        else:
            return inputs

    def backwards(self, error_signal):
        return (self.scale * error_signal) * self.activated_inputs #teoricamente si

    def update_weights (self, eta, lam, alpha): #lanciare questo significa che è finito un batch
        pass
        #self.activated_inputs = np.random.choice([1., 0.], size=(self.input_dim, ), p=[self.rate, 1-self.rate])

    def get_weights(self):
        return [0]
