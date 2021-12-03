import random

import numpy as np

from dataloader import DataLoader
import nn_unit
import MLP
import activation_functions

def MSE_over_network(batch, NN):
    errors = []
    for pattern in batch:
        out = NN.forward(pattern[0])
        errors.append((out - pattern[1])**2)
    mse = sum(errors)/len(errors)
    return mse

def pick_batch(TS, len_epoch):
    batch = random.sample(TS, len_epoch)
    return batch

def backpropagation_step(batch, NN, eta):
    for pattern in batch:
        print(f"pattern: {pattern[0]}, {pattern[1]}")
        NN.forward(pattern[0])
        current_layer_key = len(NN.layer_struct) - 1
        current_layer = NN.layer_set[current_layer_key]
        delta_up = np.array([])
        delta_curr = np.array([])
        delta_w = np.array([])
        new_weights = np.array([])
#        print(f"{current_layer.unit_set}")
        for k in range(len(current_layer.unit_set)):
            unit = current_layer.unit_set[k]
            out, out_prime = unit.get_outputs()
            #bisognerà fare pattern[1][k] se l'output è vettoriale
            delta_k = ((pattern[1] - out) * out_prime)
            delta_w = np.append(delta_w, delta_k * out)
#            np.append(delta_up, delta_k)

#        current_layer.update_unit_weights(delta_w, eta)

        new_weights = np.append(new_weights, delta_w * eta/float(len(batch)))
        print(f"{delta_w} * {eta}")
        delta_w = np.array([])
        
        current_layer_key -= 1
        current_layer = NN.layer_set[current_layer_key]            
        while current_layer.layer_prec != None:
            for k in range(len(current_layer.unit_set)):
                unit = current_layer.unit_set[k]
                out, out_prime = unit.get_outputs()
                delta_k = ((delta_up * 1).sum() * out_prime) #l'1 è temporaneo
                delta_w = np.append(delta_w, delta_k * out)
                delta_curr = np.append(delta_curr, delta_k)
                #non vi sento :P
#            current_layer.update_unit_weights(delta_w, eta)
            new_weights = np.append(new_weights, delta_w * eta/float(len(batch)))            
            current_layer_key -= 1
            current_layer = NN.layer_set[current_layer_key]  
            delta_up = delta_curr
            delta_curr = np.array([])
            delta_w = np.array([])
    print(f"new_weights: {new_weights}")
    NN.update_all_weights(new_weights)
        

def backpropagation(TS, NN, eta, len_epoch, thr=0.001, N_max=1000):
    NN.initialise_weight_matrix()
    current_batch = pick_batch(TS, len_epoch)
    Error = MSE_over_network(current_batch, NN)
    Error_series = [Error]
    i = 0
    while (Error > thr) & (i < N_max):
        current_batch = pick_batch(TS, len_epoch)
        backpropagation_step(current_batch, NN, eta)
        Error = MSE_over_network(current_batch, NN)
        Error_series.append(Error)
        i += 1
    return Error_series
