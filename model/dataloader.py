import numpy as np
import os
import random

def _1_hot_enc(inputs, domains):
    """
    inputs: list of scalars, of length n
    domains: list of domains, i.e. a list of lists such that 
        for each i, domains[i] = domain of inputs[i]
    """
    encoded = []
    for input, dom in zip(inputs, domains):
        if not (input in dom):
            raise Exception("Incorrect encoding!")
        vector = [(1 if input == val else 0) for val in dom]
        encoded.extend(vector)
    return encoded

class DataLoader():

    def __init__(self):
        self.DATA_PATH = os.path.join(".", "data")
        self.batch_size = 32
        self.data = {}

    def load_data(self, data_key, filename, encoding=None):
        self.data[data_key] = []
        full_fn = os.path.join(self.DATA_PATH, filename)
        f = open (full_fn, "r")
        for line in f.readlines():
            l = list(map(int, line.split()[:-1]))
            data = list(map(int, line.split()[:-1]))
            if encoding != None:
                inputs = _1_hot_enc(data[1:], encoding)
            else:
                inputs = data[1:]
            output = data[0]
            pattern = (np.array(inputs), np.array(output))
            self.data[data_key].append(pattern)

    def get_train_batch (self, batch_size):
        batch_data = random.choices(self.data['train'], k=batch_size)
        return batch_data

    def get_inputs_dimension(self):
        return self.data['train'].shape[1]
