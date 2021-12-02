import random

import numpy as np

from dataloader import DataLoader
import nn_unit
import MLP
import activation_functions

def MSE_over_network(batch, NN):
    errors = []
    out_layer = NN.get_output_layer()
    for unit in output_layer:
        for pattern in batch:
            errors.append((unit.out(pattern[0]) - pattern[1])**2)
    mse = sum(errors)/len(errors)
    return mse

def pick_batch(TS, len_epoch):
    batch = random.sample(TS, len_epoch)
    return batch

def backpropagation_step(batch, NN, eta):
    for pattern in batch:
        print(f"pattern: {pattern[0]}, {pattern[1]}")
        current_layer_key = len(NN.layer_struct) - 1
        current_layer = NN.layer_set[current_layer_key]
        delta_up = np.array([])
        delta_curr = np.array([])
        print(f"{current_layer.unit_set}")
        for k in range(len(current_layer.unit_set)):
            unit = current_layer.unit_set[k]
            #bisognerà fare pattern[1][k] se l'output è vettoriale
            delta_k = (pattern[1] - unit.out(pattern[0]) * unit.out_prime(pattern[0]))
            np.append(delta_up, delta_k)
            for i, weight in enumerate(unit.weights):
                delta_w = delta_up * unit.inputs[i]
                current_layer.weights[key][i] += delta_w * eta/float(len(batch))
        current_layer_key -= 1
        current_layer = NN.layer_set[current_layer_key]            
        while current_layer.layer_prec != None:
            for k in range(len(current_layer.unit_set)):
                unit = current_layer.unit_set[k]
                delta_k = (delta_up * NN.layer_set[current_layer_key+1]).sum() * unit.out_prime(pattern[:-1])
                np.append(delta_curr, delta_k)
                for i, weight in enumerate(unit.weights):
                    delta_w = delta * unit.inputs[i]
                    current_layer.weights[key][i] += delta_w * eta/float(len(batch))
            current_layer_key -= 1
            current_layer = NN.layer_set[current_layer.layer_prec]  
            delta_up = delta_curr
            delta_curr = np.array([])
    NN.update_all_weights()

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
