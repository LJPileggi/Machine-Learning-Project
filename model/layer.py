from os import error
from site import enablerlcompleter
import numpy as np
import math
import activation_functions

class Layer():
    """
    Single layer of the MLP.
    This is a superclass used to define a typical layer
    """
    
    def __init__(self):
        """ Constructor of the Layer class

        Args:

        Returns:
            A Layer Object
        """
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
        """Activation function method

        Args:
            network_value (ndarray): The network value of each node in the layer
        
        Returns:
            NotImplementedError if the subclass didn't implement this method
            out a ndarray of float, the final output of each node
        """
        raise NotImplementedError ("This layer doesn't have an activation Function")

    def activation_prime (self, network_value):
        """Activation prime function method

        Args:
            network_value (ndarray): The network value of each node in the layer
        
        Returns:
            NotImplementedError if the subclass didn't implement this method
            out_prime a ndarray of float, the output of each node given the derivative of the activation function
        """
        raise NotImplementedError ("This layer doesn't have an activation Function")

    def forward (self, inputs, training=True):
        """Forward method
        This method does the forward pass of the layer.
        The forward pass consist of calculating the output of each node given the inputs
        The output is calculated as the inputs * weights + biases
        This method works with a singular input or a batch of inputs

        Args:
            inputs (ndarray): The inputs of this layer. if ndim = 1 then it calculate the forward over a single pattern, if ndim = 2 then calculate the outputs of an entire batch of inputs 
        
        Returns:
            output (ndarray): the output vector (or matrix) of all the nodes in the layer given the inputs vector (or matrix)
        """
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

    def backwards(self, error_signal):
        """Backwards method
        This method does the backwards pass of the layer
        The backwards pass consist in calculating the delta and negative gradients of the layer given the error signal from the layer above
        the delta is calculated as error_signal * output_prime
        the negative gradient is calculated as the outer product between the inputs and the deltas
        it's returned the error signal for the next layer, calculated as deltas * weights
        This method too accept a single input or a batch of error_signals

        Args:
            error_signal (ndarray): The error_signal of the layer above. if ndim = 1 then it calculate the backwards over a single ES, if ndim = 2 then calculate the outputs of an entire batch of ES
        
        Returns:
            error_signal (ndarray): the error_signal vector (or matrix) to be backpropagated to the layers under us given the error_signal vector (or matrix) of the layer above
        """
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
        """Update weights method
            here we apply the negGradient computed during backward.
            In a online alg, we continously call this.backward and this.update_weights,
              so for each pattern, negGrad is computed and applied immediately.
            In a (mini)batch alg, we call this.backward for each pattern, and
              this.update_weights just once per batch.

            Args:
                eta (float) - the learning rate of this update
                lam (float) - the lambda used for tikhonov regularization
                alpha (float) - the alpha used to calculate the momentum
        """
        
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
        """Get Max Grad method
            A method used to get the maximum gradient it the net

            Args:

            Returns:
                last_max_grad
        """
        return self._last_max_grad

    def get_weights(self):
        """Get weights
            A method used to retrieve a list of all the weights and biases of the layer

            Returns:
                all_weights (list): a flatten ndarray with all the weights and biases of the units in this layer
        """
        return np.vstack((self._WM, self._biases)).flatten() #if (self._WM != None and self._biases != None) else [0]
        
class Sigmoidal(Layer):
    """Subclass of Layer for the Sigmoidal type
    """

    def __init__(self, input_dim, layer_dim):
        """Constructor of the Sigmoidal Layer
        In this constructor we intialize the weights and biases of the layer using a normalized uniform xavier

        Args:
            input_dim (int): the number of input of each node
            layer_dim (int): the number of nodes in this layer
        """
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
        """Constructor of the Tanh Layer
        In this constructor we intialize the weights and biases of the layer using a normalized uniform xavier

        Args:
            input_dim (int): the number of input of each node
            layer_dim (int): the number of nodes in this layer
        """
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
        """Constructor of the Linear Layer
        In this constructor we intialize the weights and biases of the layer using a normalized uniform xavier

        Args:
            input_dim (int): the number of input of each node
            layer_dim (int): the number of nodes in this layer
        """
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
        """Constructor of the ReLu Layer
        In this constructor we intialize the weights and biases of the layer using a HeNormal weights initialization

        Args:
            input_dim (int): the number of input of each node
            layer_dim (int): the number of nodes in this layer
        """
        print(f"{layer_dim}; ReLu")
        super().__init__()
        #self._WM = np.random.normal(loc=0.0, scale=math.sqrt(2/input_dim), size=(input_dim, layer_dim)) * math.sqrt(2./input_dim) #HeNormal weight initialization
        lower, upper = -math.sqrt(6/input_dim), math.sqrt(6/input_dim)
        self._WM = np.random.uniform(low=lower, high=upper, size=(input_dim, layer_dim) ) #HeUniform weight initialization
        self._biases = np.zeros(shape=(layer_dim,))

    def activation (self, network_value):
        return np.maximum(0, network_value)

    def activation_prime (self, network_value):
        return np.where(network_value > 0, 1., 0.)

class BatchNormalization(Layer):

    def __init__(self, momentum=0.99, epsilon=0.001):
        """Constructor of the BatchNormalization Layer
        In this constructor we intialize the special weights of this layer.

        Args:
            momentum (float): the momentum at wich we want to update the moving mean and var
            epsilon (int): a small number to help us stabilize the division 
        """
        print(f"BatchNormalization")
        super().__init__()
        self.gamma = 1
        self.beta = 0
        self.epsilon = epsilon #per stabilità numerica
        self.moving_mean = 0
        self.moving_var = 0
        self.momentum = momentum
        self._betaold = 0
        self._gammaold = 0

    def forward(self, inputs, training=True):
        self.inputs = inputs
        if (training):
            n, m = inputs.shape

            mean = 1./n * np.sum(inputs, axis=0) #forse axis = 1

            evil_inputs = inputs - mean

            variance = 1./n * np.sum(self.inputs**2, axis=0)

            evil_inputs = evil_inputs * (variance + self.epsilon)**(-1./2.)

            out = self.gamma * evil_inputs + self.beta

            self.moving_mean = self.moving_mean*self.momentum + mean * (1-self.momentum)
            self.moving_var = self.moving_var*self.momentum + variance * (1-self.momentum)
        else:
            evil_inputs = (inputs - self.moving_mean) * (self.moving_var + self.epsilon)**(-1./2.)
            out = self.gamma * evil_inputs + self.beta
        return out

    def backwards(self, error_signal):
        n, m = error_signal.shape

        mean = 1./n * np.sum(self.inputs, axis = 0) #usare moving mean? No perché siamo ancora con una batch
        variance = 1./n * np.sum((self.inputs - mean)**2, axis = 0)
        
        self.dbeta = np.sum(error_signal, axis=0)
        self.dgamma = np.sum(((self.inputs-mean) * (variance + self.epsilon)**(-1./2.) * error_signal), axis=0)
        new_error_signal = ((1./n) * self.gamma * (variance + self.epsilon)**(-1./2.) * (n * error_signal - np.sum(error_signal, axis=0))
                            - (self.inputs - mean) * (variance + self.epsilon)**(-1.) * np.sum(error_signal - (self.inputs - mean), axis=0))
        return new_error_signal

    def update_weights (self, eta, lam, alpha): #lanciare questo significa che è finito un batch
        B = eta * self.dbeta + alpha*self._betaold
        self.beta += B - lam*self.beta
        self._betaold = B

        G = eta * self.dgamma + alpha*self._gammaold
        self.gamma += G - lam*self.gamma
        self._gammaold = G


    def get_weights(self):
        return [0]
        

class Dropout(Layer): #potremmo benissimo trasformarlo in un layer tutto suo

    def __init__(self, rate):
        """Constructor of the Dropout Layer
        In this constructor we intialize the rate of the dropout and the scale of wich the input will be multiplied

        Args:
            rate (int): the rate at wich the nodes will be turned on (turn off at 1-rate)
        """
        print(f"Dropout: {rate}")
        super().__init__()
        if isinstance(rate, (int, float)) and not 0 <= rate <= 1:
            raise ValueError (f"Invalid Value {rate} received - `rate` needs to be betweek 0 and 1")
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
