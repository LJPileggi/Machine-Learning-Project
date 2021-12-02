import numpy as np
import os
import random

class DataLoader():

    def __init__(self):
        self.DATA_PATH = os.path.join(".", "data")
        self.batch_size = 32
        self.data = {}
        
    def load_data(self, data_key, filename):
        full_fn = os.path.join(self.DATA_PATH, filename)
        f = open (full_fn, "r")
        for line in f.readlines():
            l = list(map(int, line.split()[:-1]))
            data[data_key].append( (np.array(l[1:]), np.array(l[0])) )

    def get_train_batch (self, batch_size):
        batch_data = random.choices(len(self.data['train']), k=batch_size)
        return batch_data
