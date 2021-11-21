import random
import numpy as np

import nn_unit


"""
Funzione per impostare il seed a tutti i moduli che lo richiedono

params:
 - seed: Il seme che stiamo impostando
"""
def set_seed (seed):
    random.seed(seed)
    np.random.seed(seed)



    def backprog (self, patterns = []): #online
        if (self.error_signal == None and patterns != None): #per vedere se fare come output layer o come hidden layer.
            tmp_delta = patterns - self.out
            delta = tmp_delta * act_prime (self.net) #delta è un array di dimensione #patterns
        else:
            delta = erro_signal * act_prime (self.net)
        self.layer_prec.set_error_signal(np.sum((delta * self.weights), axis = 1)) #viene un vettore, con la somma degli elementi sulle righe della matrice weights moltiplicata a delta

        self.weights = self.weights + delta.T * self.out #vettore riga delta * vettore riga colonna fa una matrice di dimensione uguale a weights che è quello per cui dobbiamo addare
        if (layer_prec == None):
            return
        else:
            self.layer_prec.backprog() 
        
    def set_error_signal (self, error_signal):
        self.error_signal = error_signal #sono i delta backpropagati dal layer successivo

        
    def act_prime (self, network_value):
        if (self.activation == "linear"):
            return 1
        
        
"""
In questo file noi faremo il training del modello

"""
if __name__ == '__main__':


    input_l = layer (layer_prec = None,
                     input = np.zeros([x,], dtype=float),
                     net   = np.zeros([10,], dtype=float),
                     out   = np.zeros([10,], dtype=float),
                     weights=np.random.rand(10, x),
                     activation = "linear")
    hidden_l = layer (layer_prec = input_l,
                      input = np.zeros([10,], dtype=float),
                      net   = np.zeros([30,], dtype=float),
                      out   = np.zeros([30,], dtype=float),
                      weights=np.random.rand(30, 10),
                      activation = "linear")
    output_l = layer (layer_prec = hidden_l,
                      input = np.zeros([30,], dtype=float),
                      net   = np.zeros([5,], dtype=float),
                      out   = np.zeros([5,], dtype=float)
                      weights=np.random.rand(5, 30),
                      activation = "linear")

    
