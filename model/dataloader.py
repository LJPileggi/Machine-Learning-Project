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

    def load_data_from_dataset (self, filename, encoding=None, train_slice=1, k_fold=5): #implementare il kfold
        """
        set the data into the internal dictionary.
        train_slice define how much of this dataset it's going to be training

        """
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

        # forse visto che è one hot ha senso salvarsi in bool? na
        self.data["full"] = np.array(dataset, dtype=object) #cambiamo e ci salviamo tutto il dataset, che poi splitteremo usando gli indici della funzione successiva

        """
        tot_len = len(dataset)
        train_separator = int(tot_len*train_slice)
        val_separator = int(train_separator/2)
        self.data['train'] = dataset[:train_separator]
        self.data['val'] = dataset[train_separator:train_separator+val_separator]
        self.data['test'] = dataset[train_separator+val_separator:]
        """

    def get_slices (self, k_fold=5):
        if k_fold > 1: 
            n_samples = len(self.data["full"])
            indices = np.arange(n_samples)

            fold_sizes = np.full(k_fold, n_samples // k_fold, dtype=int)
            fold_sizes[: n_samples % k_fold] += 1
            current = 0
            
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                test_mask = np.zeros(n_samples, dtype=bool)
                test_mask[indices[start:stop]] = True
                train_index = indices[np.logical_not(test_mask)]
                test_index = indices[test_mask]
                yield train_index, test_index
                current = stop
        else:
            n_samples = len(self.data["full"])
            indices = np.arange(n_samples)
            train_start = n_samples // 5
            return indices[train_start:], indices[:train_start]


        
    def dataset_partition (self, indices, batch_size): #
        """
        returns an iterator on minibatches:
        if batch_size=n, returns a list of n patterns
        if batch_size=1, returns 1 pattern at a time (techincally, a list conteining just one pattern)
        after traversing the whole TS, the TS is shuffled
        """
        tr_size = len(indices)
        batch_size = tr_size if batch_size == -1 else batch_size
        curr_data = self.data["full"][indices]
        idxs = np.arange(tr_size)
        batchs_sizes = np.full(tr_size // batch_size, batch_size, dtype=int)
        batchs_sizes[: tr_size % batch_size] += 1
        current = 0
        random.shuffle(curr_data)
        
        for batch in batchs_sizes:
            start, stop = current, current + batch
            yield curr_data[start:stop]
            current = stop

    def get_input_size(self):
        return self.data['full'][0][0].size
            
    def get_partition_set(self, indices): #aggiungere parametro per il kfold
        return self.data["full"][indices]

