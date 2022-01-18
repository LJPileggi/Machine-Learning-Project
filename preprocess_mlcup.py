import os
import csv
import argparse

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
        var += vec**2/len(vec_set)
    mean /= len(vec_set)
    var -= mean**2
    return mean, var, var**0.5

def data_standardiser(data):
    mean, _, dev = mean_variance_dev(data)
    return [value-mean/dev for value in data]

def data_normaliser(dataset):
    min_data, max_data = [comp for comp in dataset[0]], [comp for comp in dataset[0]]
    for data in dataset[1:]:
        for i, component in enumerate(data):
            if component < min_data[i]:
                min_data[i] = component
            elif component > max_data[i]:
                max_data[i] = component
            else:
                pass
    min_data, max_data = np.array(min_data), np.array(max_data)
    norm = []
    for data in dataset:
        print(data)
        norm.append(data-min_data)/(max_data-min_data)
    return norm

def file_parsing():
    parser = argparse.ArgumentParser(description='Preprocessing of data for regression via normalisation or standardisation.')
    parser.add_argument('--filename', metavar='filename', type=str, dest='filename', help='Name of file to preprocess.')
    parser.add_argument('--input_task', metavar='input_task', dest='input_task', help='Select task to perform on input data; choose between "stand" and "norm". Default on normalisation.')
    parser.add_argument('--output_task', metavar='output_task', dest='output_task', help='Select task to perform on output data; choose between "stand" and "norm". Default on normalisation.')
    parser.set_defaults(input_task=None)
    parser.set_defaults(output_task=None)
    args = parser.parse_args()
    return args
    
def main():
    folder = "./data/"
    """
    title = "ML-CUP21-"
    dataset = ["TR.csv", "TS.csv"]
    filenames = [folder+title+data for data in dataset]
    """
    args = file_parsing()
    filename, input_task, output_task = args.filename, args.input_task, args.output_task
    #for filename, data in zip(filenames, dataset):
    f_in = open(folder+filename, 'r', newline='')
    f_out = open(folder+"processed-"+filename, 'w', newline='')
    data_id = []
    inputs = []
    outputs = []
    writer = csv.writer(f_out)
    for line in f_in.readlines():
        pattern = line.split()
        data_id.append(pattern[0])
        inp = np.array(pattern[1:-2])
        inputs.append(inp)
        out = np.array(pattern[-2:])
        outputs.append(out)
    if input_task == "norm":
        inputs = data_normaliser(inputs)
    elif input_task == "stand":
        inputs = data_standardiser(inputs)
    else:
        pass
    if output_task == "norm":
        outputs = data_normaliser(outputs)
    elif output_task == "stand":
        outputs = data_standardiser(outputs)
    else:
        pass
    """
    mean, var = mean_and_variance(output_row)
    mean = list(mean)
    null = list(np.zeros(10))
    writer.writerow(var+null+mean)
    """
    for id, input, output in zip(data_id, inputs, outputs):
        results = [id]
        results.extend(input)
        results.extend(output)
        writer.writerow(results)


if __name__ == '__main__':
    main()
