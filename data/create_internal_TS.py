import numpy as np
import random
import csv
import time
import argparse

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def divide_et_impera (filename, TR_fn, TS_fn, train_part=0.9):
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        data = []
        for row in reader:
            data.append(list(map(float, row)))

    train_mask = np.random.choice([True, False], size=len(data), p=[train_part, 1-train_part])
    data = np.array(data, dtype=float)
    train_data = data[train_mask]
    test_data = data[~train_mask]

    with open(TR_fn, 'w') as tr_f:
        writer = csv.writer(tr_f, delimiter=',')
        for ele in train_data:
            writer.writerow(ele)
    with open(TS_fn, 'w') as ts_f:
        writer = csv.writer(ts_f, delimiter=',')
        for ele in test_data:
            writer.writerow(ele)


set_seed(int(time.time()))
parser = argparse.ArgumentParser()
parser.add_argument('--filename',
                    help='filename to divide into a train and internal test set')
args = parser.parse_args()
divide_et_impera(args.filename, "mlcup_internaltrain.csv", "mlcup_internaltest.csv", 0.75)
