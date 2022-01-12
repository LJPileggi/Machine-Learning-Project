from dataloader import DataLoader
from datastorage import DataStorage
from learning_algs import grid_search
import random
import numpy as np
import argparse
import json
from datetime import datetime
import os
import time


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
 


# def count(dl, global_confs, local_confs, output_path, graph_path, seed=4444):
#     history = {}
#     history['mean'] = 1.
#     layers      = local_confs["layers"]
#     batch_size  = local_confs["batch_size"]
#     eta_decay   = local_confs["eta_decay"]#if == -1 no eta decay; 25 should be fine
#     eta         = local_confs["eta"]
#     lam         = local_confs["lambda"]
#     alpha       = local_confs["alpha"] 
#     history['hyperparameters'] = (layers, batch_size, eta, lam, alpha, eta_decay)
#     return history

def main():
    ### Parsing cli arguments ###
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('--config_path',
                        help='path to config file')
    parser.add_argument('--seed', type=int,
                        help='random seed')
    parser.add_argument('--loop', type=int,
                        help='how many nested loop you want to do')
    parser.add_argument('--shrink', type=float,
                        help='how much do you want to shrink during nested grid search')
    parser.set_defaults(seed=int(time.time())) #when no seed is provided in CLI nor in config, use the unix time
    parser.set_defaults(shrink=0.1)
    parser.set_defaults(loop=3)
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    ### loding hyperparameters from config ###
    hyperparameters = config["hyperparameters"]

    ### loading CONSTANT parameters from config ###
    global_conf = config["global_conf"]
    
    ### loading or generating seed ###
    seed = args.seed # prendiamo da riga di comando. il default è l'epoch time
    print(f"seed: {seed}")
    set_seed(seed)
    global_conf["seed"] = seed

    ### loading and preprocessing dataset from config ###
    dl = DataLoader(seed)
    dl.load_data(config["train_set"], config["input_size"], config["output_size"], config.get("preprocessing"))

    ### setting up output directories ###
    data_conf = config["data_conf"]
    now = datetime.now()
    date = str(datetime.date(now))
    time = str(datetime.time(now))
    time = time[:2] + time[3:5]
    
    output_path = os.path.abspath(data_conf["output_path"])
    output_path = os.path.join(output_path, date, time)
    print(output_path)
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    
    graph_path = os.path.abspath(data_conf["graph_path"])
    graph_path = os.path.join(graph_path, date, time)
    print(graph_path)
    if (not os.path.exists(graph_path)):
        os.makedirs(graph_path)
    
    #executing training and model selection
    grid_search(dl, global_conf, hyperparameters, output_path, graph_path, args.loop, args.shrink)
    
    ##here goes testing
    print("grid search complete!")



if __name__ == '__main__':
    main()
