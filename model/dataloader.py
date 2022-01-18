import numpy as np
import math
import os
import csv
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

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
    var  /= len(vec_set)
    var -= mean**2
    return mean, var, var**0.5

def data_standardizer(data):
    mean, _, dev = mean_variance_dev(data)
    return [value-mean/dev for value in data]

class DataLoader():
    
    DATA_PATH = os.path.join("..", "data")

    def __init__(self, seed=4444):
        set_seed(seed)
        self.data = {}

    def load_data (self, filename, input_size, output_size, preprocessing, tag="full", shuffle=True): #normalmente usiamo full che è anche quello che salva tutto e poi verrà usato pe kfold. Oppure possiamo usare train e test
        """
        set the data into the internal dictionary.
        train_slice define how much of this dataset it's going to be training

        """
        full_fn = os.path.join(self.DATA_PATH, filename)
        with open(full_fn) as f:
            reader = csv.reader(f, delimiter=',')
            inputs = []
            outputs = []
            #reads file
            for row in reader:
                if len(row) == 1 + input_size + output_size:
                    data = list(map(float, row[1:]))
                    input = data[:-output_size]
                    output = data[-output_size:]
                    inputs.append(np.array(input))
                    outputs.append(np.array(output))
                else:
                    raise ValueError(f"wrong input or output sizes at line {reader.line_num}")
            
            #preprocesses data, if needed
            if(preprocessing == None):
                pass
            elif(preprocessing == "output_stand"):
                outputs = data_standardizer(outputs)
            elif(preprocessing == "input_stand"):
                inputs = data_standardizer(inputs)
            elif(preprocessing == "both_stand"):
                inputs = data_standardizer(inputs)
                outputs = data_standardizer(outputs)
            else:
                raise NotImplementedError("unknown preprocessing")

            #save data
            dataset = list(zip(inputs, outputs))
            if (shuffle):
                np.random.shuffle(dataset)
            self.data[tag] = np.array(dataset, dtype=object) #cambiamo e ci salviamo tutto il dataset, che poi splitteremo usando gli indici della funzione successiva
            
    @staticmethod
    def load_data_static (filename, input_size, output_size, preprocessing, shuffle=True): #normalmente usiamo full che è anche quello che salva tutto e poi verrà usato pe kfold. Oppure possiamo usare train e test
        """
        set the data into the internal dictionary.
        train_slice define how much of this dataset it's going to be training

        """
        full_fn = os.path.join( DataLoader.DATA_PATH, filename)
        with open(full_fn) as f:
            reader = csv.reader(f, delimiter=',')
            inputs = []
            outputs = []
            #reads file
            for row in reader:
                if len(row) == 1 + input_size + output_size:
                    data = list(map(float, row[1:]))
                    input = data[:-output_size]
                    output = data[-output_size:]
                    inputs.append(np.array(input))
                    outputs.append(np.array(output))
                else:
                    raise ValueError(f"wrong input or output sizes at line {reader.line_num}")
            
            #preprocesses data, if needed
            if(preprocessing == None):
                pass
            elif(preprocessing == "output_stand"):
                outputs = data_standardizer(outputs)
            elif(preprocessing == "input_stand"):
                inputs = data_standardizer(inputs)
            elif(preprocessing == "both_stand"):
                inputs = data_standardizer(inputs)
                outputs = data_standardizer(outputs)
            else:
                raise NotImplementedError("unknown preprocessing")

            #save data
            dataset = list(zip(inputs, outputs))
            if (shuffle):
                np.random.shuffle(dataset)
            return np.array(dataset, dtype=object) #cambiamo e ci salviamo tutto il dataset, che poi splitteremo usando gli indici della funzione successiv

    def get_slices (self, k_fold=5, tag='full'):
        if k_fold > 1: 
            n_samples = len(self.data[tag])
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
            n_samples = len(self.data[tag])
            indices = np.arange(n_samples)
            train_start = n_samples // 5
            yield indices[train_start:], indices[:train_start]

    @staticmethod
    def get_slices_static (data, k_fold=5):
        n_samples = len(data)
        if k_fold > 1:
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
                yield data[train_index], data[test_index]
                current = stop
        elif k_fold == 1:
            indices = np.arange(n_samples)
            train_start = n_samples // 5
            yield data[indices[train_start:]], data[indices[:train_start]]
        else:
            yield data, None

     
    def dataset_partition (self, indices, batch_size, tag='full'): #
        """
        returns an iterator on minibatches:
        if batch_size=n, returns a list of n patterns
        if batch_size=1, returns 1 pattern at a time (techincally, a list conteining just one pattern)
        after traversing the whole TS, the TS is shuffled
        """
        tr_size = len(indices)
        batch_size = tr_size if batch_size == -1 else batch_size
        curr_data = self.data[tag][indices]
        idxs = np.arange(tr_size)
        batchs_sizes = np.full(tr_size // batch_size, batch_size, dtype=int)
        batchs_sizes[: tr_size % batch_size] += 1
        current = 0
        np.random.shuffle(curr_data)
        
        for batch in batchs_sizes:
            start, stop = current, current + batch
            yield curr_data[start:stop]
            current = stop

    @staticmethod
    def dataset_partition_static (data, batch_size, shuffle=True): #
        """
        returns an iterator on minibatches:
        if batch_size=n, returns a list of n patterns
        if batch_size=1, returns 1 pattern at a time (techincally, a list conteining just one pattern)
        after traversing the whole TS, the TS is shuffled
        """
        tr_size = len(data)
        batch_size = tr_size if batch_size == -1 else batch_size
        idxs = np.arange(tr_size)
        batchs_sizes = np.full(tr_size // batch_size, batch_size, dtype=int)
        batchs_sizes[: tr_size % batch_size] += 1
        current = 0
        if (shuffle):
            np.random.shuffle(data)
        
        for batch in batchs_sizes:
            start, stop = current, current + batch
            yield data[start:stop]
            current = stop

    def get_input_size(self, tag='full'):
        return self.data[tag][0][0].size

    @staticmethod
    def get_input_size_static(data):
        return data[0][0].size
            
    def get_partition_set(self, indices, tag='full'):
        return self.data[tag][indices]
    
    def get_tag_set(self, tag): #aggiungere parametro per il kfold
        return self.data[tag]

# class MonkDataLoader(AbstractDataLoader):
#     def __init__(self):
#         super().__init__()
    
#     def load_data (self, filename, encoding=None, train_slice=1, k_fold=5): #implementare il kfold
#         """
#         set the data into the internal dictionary.
#         train_slice define how much of this dataset it's going to be training

#         """
#         full_fn = os.path.join(self.DATA_PATH, filename)
#         f = open (full_fn, "r")
#         dataset = []
#         for line in f.readlines():
#             data = list(map(int, line.split()[:-1]))
#             if encoding != None:
#                 inputs = _1_hot_enc(data[1:], encoding)
#             else:
#                 inputs = data[1:]
#             output = data[0]
#             pattern = (np.array(inputs), np.array(output))
#             dataset.append(pattern)
#         random.shuffle(dataset)

#         # forse visto che è one hot ha senso salvarsi in bool? na
#         self.data["full"] = np.array(dataset, dtype=object) #cambiamo e ci salviamo tutto il dataset, che poi splitteremo usando gli indici della funzione successiva

#     # def load_data(self, data_key, filename, encoding=None):
#     #     self.data[data_key] = []
#     #     full_fn = os.path.join(self.DATA_PATH, filename)
#     #     f = open (full_fn, "r")
#     #     for line in f.readlines():
#     #         data = list(map(int, line.split()[:-1]))
#     #         if encoding != None:
#     #             inputs = _1_hot_enc(data[1:], encoding)
#     #         else:
#     #             inputs = data[1:]
#     #         output = data[0]
#     #         pattern = (np.array(inputs), np.array(output))
#     #         self.data[data_key].append(pattern)


# class MLCupDataLoader(AbstractDataLoader):
#     def __init__(self):
#         super().__init__()
    
#     def load_data (self, filename, encoding=None, train_slice=1, k_fold=5): #implementare il kfold
#         """
#         set the data into the internal dictionary.
#         train_slice define how much of this dataset it's going to be training

#         """
#         full_fn = os.path.join(self.DATA_PATH, filename)
#         with open(full_fn) as f:
#             reader = csv.reader(f, delimiter=',')
#             dataset = []
#             for row in reader:
#                 data = list(map(float, row[1:]))
#                 inputs = data[:-2]
#                 output = data[-2:]
#                 pattern = (np.array(inputs), np.array(output))
#                 dataset.append(pattern)
#             random.shuffle(dataset)

            
#             self.data["full"] = np.array(dataset, dtype=object) #cambiamo e ci salviamo tutto il dataset, che poi splitteremo usando gli indici della funzione successiva
