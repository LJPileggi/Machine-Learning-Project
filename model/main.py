from matplotlib.pyplot import grid
from dataloader import DataLoader
from MLP import MLP
from postprocess import MSE_over_network, empirical_error, create_graph
from learning_algs import grid_search
from multiprocessing import Pool
from datetime import datetime
import random
import numpy as np
import os
import argparse
import json
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
    parser.add_argument('--seed',
                        help='random seed')
    parser.add_argument('--nested', dest='nested', action='store_true',
                        help='if you want to do a nested grid search')
    parser.add_argument('--shrink',
                        help='how much do you want to shrink during nested grid search')
    parser.add_argument('--loop',
                        help='how many nested loop you want to do')
    parser.set_defaults(seed=int(time.time())) #when no seed is provided in CLI nor in config, use the unix time
    parser.set_defaults(nested=False)
    parser.set_defaults(shrink=0.1)
    parser.set_defaults(loop=3)
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    ### setting up output directories ###
    now = datetime.now()
    now_date = str(datetime.date(now))
    now_time = str(datetime.time(now))
    now_time = now_time[:2] + now_time[3:5]
    output_path = os.path.abspath(config["output_path"])
    output_path = os.path.join(output_path, now_date, now_time)
    print(output_path)
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    graph_path = os.path.abspath(config["graph_path"])
    graph_path = os.path.join(graph_path, now_date, now_time)
    print(graph_path)
    if (not os.path.exists(graph_path)):
        os.makedirs(graph_path)

    ### loading or generating seed ###
    seed = int(config.get("seed", args.seed)) #prendiamo dal file di config, e se non c'è prendiamo da riga di comando. il default è l'epoch time
    print(f"seed: {seed}")
    set_seed(seed)

    ### loading and preprocessing dataset from config ###
    dl = DataLoader(seed)
    dl.load_data(config["train_set"], config["input_size"], config["output_size"], config.get("preprocessing"))

    ### loading CONSTANT parameters from config ###
    global_conf = config["model"]["global_conf"]

    ### loding hyperparameters from config ###
    hyperparameters = config["model"]["hyperparameters"]
    
    grid_search(seed, dl, global_conf, hyperparameters, bool(args.nested), int(args.loop), float(args.shrink))
    
    ##here goes model selectiom
    print("grid search complete!")



if __name__ == '__main__':
    main()
