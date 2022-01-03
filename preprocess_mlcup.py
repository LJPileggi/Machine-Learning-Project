import os
import csv

import numpy as np

def mean_variance_dev(vec_set):
    """
    Returns mean, variance and std deviation of a set
    of vectors, component wise. Returns 3 vectors
    """
    mean = 0.
    var = 0.
    for vec in vec_set:
        mean += vec
        var += vec**2
    mean /= len(vec_set)
    var -= mean**2
    return mean, var, var**0.5

def data_normaliser(data):
    mean, _, dev = mean_variance_dev(data)
    return [value-mean/dev for value in data]


if __name__ == '__main__':
    folder = "./data/"
    title = "ML-CUP21-"
    dataset = ["TR.csv", "TS.csv"]
    filenames = [folder+title+data for data in dataset]
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
