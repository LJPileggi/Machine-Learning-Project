import os
import csv
from matplotlib import pyplot as plt
import numpy as np

full_fn = "./data/mlcup_internaltrain.csv"
with open(full_fn) as f:
    output_size = 2
    reader = csv.reader(f, delimiter=',')
    #reads file
    inputs = []
    outputs = []
    for row in reader:
        data = list(map(float, row[1:]))
        input = data[:-output_size]
        output = data[-output_size:]
        inputs.append(input)
        outputs.append(output)
    
    for output_x, output_y in outputs:
        plt.plot(output_x, output_y, '.')
    plt.show()
    plt.clf()

    # for i in range(2):
    #     ith_component = [output[i] for output in outputs]
    #     plt.hist(ith_component)
    #     plt.title(f"output_{i}")
    #     plt.show()
    #     plt.clf()
    
    # for i in range(10):
    #     ith_component = [input[i] for input in inputs]
    #     plt.hist(ith_component)
    #     plt.title(f"input_{i}")
    #     plt.show()
    #     plt.clf()
    