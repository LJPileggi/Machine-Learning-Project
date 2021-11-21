#pylint: disable = C0114, C0103

import numpy as np

class nn_unit:
    """Object class for the NN unit
    params:
     - weights: weight vector; class type: numpy.ndarray;
     - activation: activation function; class type: function.

    properties:
     - network_step: function that calculate the scalar product between the input vector and the weights; class type: float;
     - output: output vector; class type: float or numpy.ndarray.
    """
    typecode = 'f'
    def __init__(self, x, weights, eta, activation_function):
        if not isinstance(type(w_i), np.ndarray):
            raise TypeError(f'TypeError: argument w_i must be <{np.ndarray}>, not <{type(w_i)}>')
        else:
            self.weights = weights
        self.eta = eta
        self.activation = activation_function #Also, questo metodo conta solo per funzioni di attivazioni senza extra parametri oltre al net_value. Consiglerei di fare una stringa e poi qui all'interno viene istanziata la funzione

    @property
    def network_step(self, input_vector):
        """net_i
        returns argument for activation function f. Type: float
        """
        return float((input_vector * self.weights).sum())

    @property
    def output(self, input_vector):
        """o_i
        returns output units. Type: either float or np.ndarray
        """
        return self.activation(input_vector)

    #proposta: mergiamo le due funzioni!

    """
    Funzione di backward dove si fa la backpropagation

    params:
     - error_signal: l'errore calcolato, equivalente a (d - o) per output units, e np.sum(delta * weights)_j per hidden unit
    return:
     - delta * weights di questa unità
    """
    @property
    def backward(self, error_signal):

        delta = error_signal * out_prime (self.network_value)

        back_signal = delta * self.weights
        
        self.weights = self.weights + self.eta * delta * self.output
        return back_signal

    @property
    def output_mike (self, input_vector):
        """
        output_mike
        versione alternativa in cui le due funzioni sopra diventano un unica funzione.
        in questa versione presniamo in considerazione che la funzione venga passata come stringa, quindi activation_function = "linear"
        returns the unit's output. Type: float or np.ndarray
        """
        self.network_value = (input_vector * self.weights).sum()
        if (activation_function == "linear"):
            self.output = linear(network_value
        elif(activation_function == "threshold"):
            self.output = threshold(network_value)
        return self.output
