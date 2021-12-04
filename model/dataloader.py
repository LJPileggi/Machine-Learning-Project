import numpy as np
import math
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
            data = list(map(int, line.split()[:-1]))
            if encoding != None:
                inputs = _1_hot_enc(data[1:], encoding)
            else:
                inputs = data[1:]
            output = data[0]
            pattern = (np.array(inputs), np.array(output))
            self.data[data_key].append(pattern)

    def training_set_partition (self, batch_size):
        """
        returns an iterator on minibatches:
        if batch_size=n, returns a list of n patterns
        if batch_size=1, returns 1 pattern at a time (techincally, a list conteining just one pattern)
        after traversing the whole TS, the TS is shuffled
        """
        tr_size = len(self.data['train'])
        batch_num = math.ceil(tr_size / batch_size) 
        random.shuffle(self.data['train'])
        for i in range(batch_num):
            yield self.data['train'][i*batch_size:(i+1)*batch_size]

    def get_training_set(self):
        return self.data['train']

    def get_test_set(self):
        return self.data['test']
