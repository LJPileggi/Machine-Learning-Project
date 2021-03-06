import signal

import joblib
from dataloader import DataLoader
from learning_algs import grid_search, train
from types import SimpleNamespace
from history import empirical_error
import random
import numpy as np
import argparse
import json
from datetime import datetime
import os
import csv
import time
from multiprocessing import set_start_method


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    os._exit(1)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    ### Parsing cli arguments ###
    parser = argparse.ArgumentParser(description="Train or Test a model.")
    parser.add_argument('--config_path',
                        help='path to config file')
    parser.add_argument('--train', action='store_true',
                        help='If you want to train the model')
    parser.add_argument('--test', action='store_true',
                        help='If you want to test the model')
    parser.add_argument('--traintest', action='store_true',
                        help='If you want to train and then test the model')
    parser.add_argument('--publish', action='store_true',
                        help='If you want to publish your result using the blind test set')
    parser.add_argument('--model_path', 
                        help='The trained model to test')
    parser.add_argument('--seed', type=int,
                        help='random seed')
    parser.add_argument('--loop', type=int,
                        help='how many nested loop you want to do')
    parser.add_argument('--shrink', type=float,
                        help='how much do you want to shrink during nested grid search')
    parser.add_argument('--preprocessing',
                        help='Selects type of preprocessing')
    parser.set_defaults(seed=int(time.time())) #when no seed is provided in CLI nor in config, use the unix time
    parser.set_defaults(train=False)
    parser.set_defaults(test=False)
    parser.set_defaults(traintest=False)
    parser.set_defaults(publish=False)
    parser.set_defaults(shrink=1)
    parser.set_defaults(loop=1)
    parser.set_defaults(preprocessing=None)
    args = parser.parse_args()
    config = SimpleNamespace(**json.load(open(args.config_path)))

    ### loding hyperparameters from config ###
    hyperparameters = config.hyperparameters

    ### loading CONSTANT parameters from config ###
    global_conf = SimpleNamespace(**config.global_conf)
    
    ### loading or generating seed ###
    seed = args.seed # prendiamo da riga di comando. il default ?? l'epoch time
    print(f"seed: {seed}")
    set_seed(seed)
    global_conf.seed = seed

    ### loading and preprocessing dataset from config ###
    #dl = DataLoader(seed)
    #dl.load_data(config["test_set"], config["input_size"], config["output_size"], config.get("preprocessing"))
    #dl.load_data(config["blind_set"], config["input_size"], config["output_size"], config.get("preprocessing"))
    TR, preproc = DataLoader.load_data_static(config.data_conf["train_set"], config.data_conf["input_size"], config.data_conf["output_size"], config.data_conf.get("preprocessing"))
    TS, _ = DataLoader.load_data_static(config.data_conf["test_set"], config.data_conf["input_size"], config.data_conf["output_size"], None)

    ### setting up output directories ###
    now = datetime.now()
    date = str(datetime.date(now))
    hour = str(datetime.time(now))
    hour = hour[:2] + hour[3:5]
    
    output_path = os.path.abspath(config.data_conf["output_path"])
    output_path = os.path.join(output_path, date, hour)
    print(output_path)
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    
    graph_path = os.path.abspath(config.data_conf["graph_path"])
    graph_path = os.path.join(graph_path, date, hour)
    print(graph_path)
    if (not os.path.exists(graph_path)):
        os.makedirs(graph_path)

    if (args.train or args.traintest):
        #executing training and model selection
        best_hyper = grid_search(TR, TS, global_conf, hyperparameters, output_path, graph_path, preproc, args.loop, args.shrink)
        print("grid search complete!")
    if (args.test or args.traintest):
        #obtaining the model
        if (args.model_path != None):
            nn = joblib.load(args.model_path)
        else:
            if (not args.traintest):
                best_hyper = hyperparameters
            nn = train(seed, config["input_size"], TR, None, TS, global_conf, best_hyper, preproc)

        #publishing the model or simply evaluating on the test set
        if (args.publish):
            BS, _ = DataLoader.load_data_static(config.data_conf["blind_set"], config.data_conf["input_size"], 0, config.data_conf.get("preprocessing"), shuffle=False)
            with open(os.path.join(output_path, "results.csv"), 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                for i, (_, inp) in enumerate(BS):
                    out = nn.predict(inp)
                    result = [i]
                    result.extend(out)
                    writer.writerow(result)
        else:
            ts_err = empirical_error(nn, TS, 'mee')
            print(ts_err)

if __name__ == '__main__':
    set_start_method('spawn')
    main()
