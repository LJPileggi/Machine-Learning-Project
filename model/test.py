#file dove loadare un modello, una config, (che ha dentro la location del file di test), e che calcoli i valori di ogni elemento

from dataloader import DataLoader
from postprocess import MSE_over_network, empirical_error

import random
import os
import argparse
import json
import time

import numpy as np

#it reads as nonlocal variables: dl, activation, checkstep, maxstep, epsilon
def train(dl, global_confs, local_confs, output_path, graph_path, seed=4444):
    try:
        #accessing data
        input_size = dl.get_input_size ()
#        whole_TR = dl.get_partition_set('train') #ritornano in auge
#        whole_VL = dl.get_partition_set('val')
        
        #set global configurations#
        task        = global_confs["task"]#either "regression" or "classification"
        max_step    = global_confs["max_step"]
        check_step  = global_confs["check_step"]
        metric      = global_confs["validation_metric"]
        epsilon     = global_confs["epsilon"]
        max_fold    = global_confs["max_fold"]
        patience    = global_confs["patience"]
        threshold   = global_confs["wc_threshold"]

        #set local configuration
        layers      = local_confs["layers"]
        batch_size  = local_confs["batch_size"]
        eta_decay   = local_confs["eta_decay"]#if == -1 no eta decay; 25 should be fine
        eta         = local_confs["eta"]
        lam         = local_confs["lambda"]
        alpha       = local_confs["alpha"]
        
        #setting history to store plots data
        history = {}
        history['training'] = []
        history['validation'] = []
        history['weight_changes'] = []
        history['testing'] = 0.
        history['val_step'] = check_step
        if eta_decay == -1:
            history['name'] = f"{layers}_{batch_size}_{eta}_nonvar_{lam}_{alpha}"
        else:
            history['name'] = f"{layers}_{batch_size}_{eta}_{eta_decay}_{lam}_{alpha}"
        history['hyperparameters'] = (layers, batch_size, eta, lam, alpha, eta_decay)  
        
        #prepares variables used in epochs#
        train_err = np.inf
        val_err = np.inf
        # old_val_err = np.inf
        # val_err_plateau = 1 #a "size 1 plateau" is just one point

        #fare un for max_fold, e per ogni fold, recuperare il whole_TR, whole_VR ecc. Poi si prendono le medie del testing e si printano i grafici 
        #create mlp#
        nn = MLP (task, input_size, layers, seed)
        oldWeights = nn.get_weights()

        #accessing the data of the k_fold
        whole_TR = dl.get_partition_set (train_idx)
        whole_VL = dl.get_partition_set (test_idx)
            
        print(f"partito un ciclo di cross val - {n_fold}")
            
        train_err = MSE_over_network (whole_TR, nn)
        history['training'].append(train_err)
        val_err = MSE_over_network (whole_VL, nn)
        history['validation'].append(train_err)
        history['weight_changes'].append(0.)

        low_wc = 0
        for i in range (max_step):
            for current_batch in dl.dataset_partition(train_idx, batch_size):
                for pattern in current_batch:
                    out = nn.forward(pattern[0])
                    error = pattern[1] - out
                    nn.backwards(error)
                    #we are updating with eta/TS_size in order to compute LMS, not simply LS
                len_batch = len(whole_TR) #if batch_size != 1 else len(whole_TR)
                if eta_decay == -1:
                    nn.update_weights(eta/len_batch, lam, alpha)
                else:
                    nn.update_weights((0.9*np.exp(-(i)/eta_decay)+0.1)*eta/len_batch, lam, alpha)
            #after each epoch
            train_err = MSE_over_network (whole_TR, nn)
            history['training'].append(train_err)
            #for layer, grad in enumerate(nn.get_max_grad_list()):
            #    history['gradients'][n_fold][layer].append(grad)
            if(i % check_step == 0):
                #once each check_step epoch
                #compute store and print validation error
                val_err = empirical_error(whole_VL, nn, metric)
                history['validation'].append(val_err)
                print (f"{n_fold}_fold - {i} - {history['name']}: {train_err} - {val_err}")
                #compute store and print weights change
                wc = 0
                newWeights = nn.get_weights()
                wc = np.mean(np.abs((oldWeights-newWeights)/oldWeights))
                oldWeights = newWeights
                history['weight_changes'].append(wc)
                #stopping criteria
                if wc <= threshold:
                    low_wc +=1
                else:
                    low_wc = 0 
                if (np.allclose(val_err, 0, atol=epsilon) or low_wc >= patience):
                    break
        
        history['testing'] = empirical_error (whole_VL, nn, metric)
        print(f"accuracy - {history['name']}: {(history['testing'][n_fold])}")

        ### saving model and plotting loss ###
        nn.save_model(os.path.join(output_path, f"model_{history['name']}_{n_fold}fold.h5"))

        ### plotting loss ###
        create_graph(history, graph_path, f"training_loss_{history['name']}.png")
        return history
    except KeyboardInterrupt:
        print('Interrupted')
        return None

def main():
    ### Parsing cli arguments ###
    parser = argparse.ArgumentParser(description="Test a model.")
    parser.add_argument('--config_path',
                        help='path to config file')
    parser.add_argument('--model_path',
                        help='path to a model to ')
    parser.add_argument('--seed',
                        help='random seed')
    parser.add_argument('--retraining', action='store_true',
                        help='if you want to retrain the model with all the train set after a successfull kfold-crossval')
    parser.add_argument('--publish', action='store_true',
                        help='if you want to generate a csv file with the results and dont confront them')
    parser.set_defaults(seed=int(time.time()))
    parser.set_defaults(retraining=False)
    parser.set_defaults(publish=False)
    parser.set_defaults(config_path=None)
    parser.set_defaults(model_path=None)
    args = parser.parse_args()
    if (args.config_path != None):
        config = json.load(open(args.config_path))

        seed = int(config.get("seed", args.seed)) #prendiamo dal file di config, e se non c'è prendiamo da riga di comando. il default è 2021
        print(f"seed: {seed}")
        set_seed(seed)
    
        dl = DataLoader (seed)

        if (retraining):
            output_path = os.path.abspath(config["output_path"])
            output_path = os.path.join(output_path, 'FINAL_MODEL')
            print(output_path)
            if (not os.path.exists(output_path)):
                os.makedirs(output_path)
            graph_path = os.path.abspath(config["graph_path"])
            graph_path = os.path.join(graph_path, 'FINAL_MODEL')
            print(graph_path)
            if (not os.path.exists(graph_path)):
                os.makedirs(graph_path)
                
            dl.load_data(config_["train_set"], config_["input_size"], config["output_size"], config.get("preprocessing"))
            ### loading CONSTANT parameters from config ###
            global_conf = config["model"]["global_conf"]

            ### loding hyperparameters from config ###
            hyperparameters = config["model"]["hyperparameters"]
            #each configuration is a triple: datas, global confs and local confs
            retrain(dl, global_conf, hyperparameters, output_path, graph_path, seed)
            

        dl.load_data(config_["test_set"], config_["input_size"], config["output_size"], config.get("preprocessing"))
