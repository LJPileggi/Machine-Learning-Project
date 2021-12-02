from dataloader import DataLoader
from MLP import MLP
import backpropagation

import numpy as np
import os
import argparse
import json

parser = argparse.ArgumentParser(description="Train a model.")
parser.add_argument('--config_path',
                    help='path to config file')

if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    train_set  = config["train_set"]
    test_set   = config["test_set"]
    model_conf = config["model"]
    
    batch_size = model_conf["batch_size"]
    epsilon    = model_conf["epsilon"]
    eta        = model_conf["eta"]
    max_step   = model_conf["max_step"]
    check_step = model_conf["check_step"]

    nn = MLP ([10, 20, 4], ["linear", "linear", "linear"])
    dl = DataLoader ()

    dl.load_data ("train", train_set)
    dl.load_data ("test", test_set)
    err = np.inf
    train_err = []

    for i in range (max_step):
        current_batch = dl.get_train_batch(batch_size)
        backpropagation.backpropagation_step(current_batch, nn, eta)
        err = MSE_over_network (current_batch, nn)
        train_err.append(Error)
        if (abs(err) > epsilon):
            break

    print(f"train_err: {train_err}")
