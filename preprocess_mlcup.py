import os
import csv

import numpy as np

def mean_and_variance(vec_set):
    mean = np.zeros(len(vec_set))
    var = 0.
    for vec in vec_set:
        mean += vec
        var += vec**2.sum()
    mean /= len(vec_set)
    var -= mean**2.sum()
    return mean, var

def data_normaliser(outputs):
    mean, dev = mean_and_variance(outputs)
    dev = dev**0.5
    Z_vec_set = []
    for output in outputs:
        Z_vec_set.append((output-mean)/dev)
    return Z_vec_set

if __name__ = '__main__':
    folder = "./data/"
    title = "ML-CUP21-"
    dataset = ["TR.csv", "TS.csv"]
    filenames = [folder+title+dataset for data in dataset]
    for filename, data in zip(filenames, dataset):
        f_in = open(filename, 'r', newline='')
        f_out = open(folder+"processed-"+title+data, 'w', newline='')
        data_id = []
        inputs = []
        outputs_row = []
        writer = csv.writer(f_out)
        for line in f_in.readlines():
            pattern = line.split()
            data_id.append(pattern[0])
            inputs.append(pattern[1:-2])
            out = np.array(pattern[-2])
            outputs_row.append(out)
        outputs = datanormaliser(outputs_row)
        mean, var = mean_and_variance(output_row)
        mean = list(mean)
        null = list(np.zeros(10))
        writer.writerow(var+null+mean)
        for id, input, output in zip(data_id, inputs, outputs):
            output = list(output)
            writer.writerow(id+input+output)
