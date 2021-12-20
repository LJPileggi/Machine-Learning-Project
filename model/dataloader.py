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
        self.DATA_PATH = os.path.join("..", "data")
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

    def load_data_from_dataset (self, filename, encoding=None, train_slice=1): #implementare il kfold
        full_fn = os.path.join(self.DATA_PATH, filename)
        f = open (full_fn, "r")
        dataset = []
        for line in f.readlines():
            data = list(map(int, line.split()[:-1]))
            if encoding != None:
                inputs = _1_hot_enc(data[1:], encoding)
            else:
                inputs = data[1:]
            output = data[0]
            pattern = (np.array(inputs), np.array(output))
            dataset.append(pattern)
        random.shuffle(dataset)
        tot_len = len(dataset)
        train_separator = int(tot_len*train_slice)
        val_separator = int(train_separator/2)
        self.data['train'] = dataset[:train_separator]
        self.data['val'] = dataset[train_separator:train_separator+val_separator]
        self.data['test'] = dataset[train_separator+val_separator:]

    def dataset_partition (self, datakey, batch_size): #aggiungere un parametro che indica che kfold volere
        """
        returns an iterator on minibatches:
        if batch_size=n, returns a list of n patterns
        if batch_size=1, returns 1 pattern at a time (techincally, a list conteining just one pattern)
        after traversing the whole TS, the TS is shuffled
        """
        tr_size = len(self.data[datakey])
        batch_num = math.ceil(tr_size / batch_size) 
        random.shuffle(self.data[datakey])
        for i in range(batch_num):
            yield self.data[datakey][i*batch_size:(i+1)*batch_size]

    def get_input_size(self):
        return len(self.data['train'][0][0])
            
    def get_partition_set(self, datakey): #aggiungere parametro per il kfold
        return self.data[datakey]

