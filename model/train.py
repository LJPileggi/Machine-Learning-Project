from dataloader import DataLoader
from MLP import MLP
from multiprocessing import Pool
from datetime import datetime
import random
import numpy as np
import os
import argparse
import json
import heapq
import time
import matplotlib.pyplot as plt

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# used in training: computes NN.forward
def MSE_over_network(batch, NN):
    mse = 0
    for pattern in batch:
        out = NN.forward(pattern[0])
        # \sum_i x_i^2, which is the square of the 2-norm
        mse += ((out - pattern[1])**2).sum() 
    return mse/len(batch)

# used in validation: computes NN.h (i.e. NN.forward > NN.threshold)
def empirical_error(batch, NN, metric):
    error = 0
    if metric == "missclass":
        for pattern in batch:
            error += np.max(abs(NN.h(pattern[0]) - pattern[1]))
    elif metric == "mse":
        for pattern in batch:
            error += ((NN.h(pattern[0]) - pattern[1])**2).sum() 
    elif metric == "mee":
        for pattern in batch:
            error += ((NN.h(pattern[0]) - pattern[1])**2).sum()**1/2
    else:
        raise NotImplementedError("unknown metric")
    return error/len(batch)

# def MEE_over_network(batch, NN):
#     mee = 0
#     for pattern in batch:
#         out = NN.h(pattern[0])
#         # \root of sum_i x_i^2, which is the euclidian norm
#         mee += ((out - pattern[1])**2).sum()**1/2
#     return mee/len(batch)

# def Missclassified_patter_rateo(batch, NN):
#     rateo = 0
#     for pattern in batch:
#         out = NN.h(pattern[0])
#         # \root of sum_i x_i^2, which is the euclidian norm
#         rateo += (out - pattern[1])
#     return rateo/len(batch)


#it reads as nonlocal variables: dl, activation, checkstep, maxstep, epsilon
def train(dl, global_confs, local_confs, output_path, graph_path, seed=4444):
    try:
        #accessing data
        input_size = dl.get_input_size ()
#        whole_TR = dl.get_partition_set('train')
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
        history['training'] = [ [] for n in range(max_fold)]
        history['validation'] = [ [] for n in range(max_fold)]
        history['weight_changes'] = [ []  for n in range(max_fold)]
        #history['gradients'] = [ [ [] for layer in layers ]  for n in range(max_fold)]
        history['testing'] = [ 0. for n in range(max_fold)]
        history['val_step'] = check_step
        if eta_decay == -1:
            history['name'] = f"{layers}_{batch_size}_{eta}_nonvar_{lam}_{alpha}"
        else:
            history['name'] = f"{layers}_{batch_size}_{eta}_{eta_decay}_{lam}_{alpha}"
        history['hyperparameters'] = (layers, batch_size, eta, lam, alpha, eta_decay)
        history['mean']      = 0
        history['variance']  = 0
        
        
        #prepares variables used in epochs#
        train_err = np.inf
        val_err = np.inf
        # old_val_err = np.inf
        # val_err_plateau = 1 #a "size 1 plateau" is just one point

        #fare un for max_fold, e per ogni fold, recuperare il whole_TR, whole_VR ecc. Poi si prendono le medie del testing e si printano i grafici di tutti.
        for n_fold, (train_idx, test_idx) in enumerate (dl.get_slices(max_fold)):
            #create mlp#
            nn = MLP (task, input_size, layers, seed)
            oldWeights = nn.get_weights()

            #accessing the data of the k_fold
            whole_TR = dl.get_partition_set (train_idx)
            whole_VL = dl.get_partition_set (test_idx)
            
            print(f"partito un ciclo di cross val - {n_fold}")
            
            train_err = MSE_over_network (whole_TR, nn)
            history['training'][n_fold].append(train_err)
            val_err = MSE_over_network (whole_VL, nn)
            history['validation'][n_fold].append(train_err)
            history['weight_changes'][n_fold].append(0.)

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
                history['training'][n_fold].append(train_err)
                #for layer, grad in enumerate(nn.get_max_grad_list()):
                #    history['gradients'][n_fold][layer].append(grad)
                if(i % check_step == 0):
                    #once each check_step epoch
                    #compute store and print validation error
                    val_err = empirical_error(whole_VL, nn, metric)
                    history['validation'][n_fold].append(val_err)
                    print (f"{n_fold}_fold - {i} - {history['name']}: {train_err} - {val_err}")
                    #compute store and print weights change
                    wc = 0
                    newWeights = nn.get_weights()
                    wc = np.mean(np.abs((oldWeights-newWeights)/oldWeights))
                    oldWeights = newWeights
                    history['weight_changes'][n_fold].append(wc)
                    #stopping criteria
                    if wc <= threshold:
                        low_wc +=1
                    else:
                        low_wc = 0 
                    if (np.allclose(val_err, 0, atol=epsilon) or low_wc >= patience):
                        break
        
            history['testing'][n_fold] = empirical_error (whole_VL, nn, metric)
            print(history['testing'][n_fold])
            history['mean'] += history['testing'][n_fold]/max_fold
            history['variance'] += history['testing'][n_fold]**2 / max_fold
            print(f"accuracy - {history['name']}: {(history['testing'][n_fold])}")

            ### saving model and plotting loss ###
            nn.save_model(os.path.join(output_path, f"model_{history['name']}_{n_fold}fold.h5"))

        history ['variance'] -= history['mean']**2
        ### plotting loss ###
        create_graph(history, graph_path, f"training_loss_{history['name']}.png")
        return history
    except KeyboardInterrupt:
        print('Interrupted')
        return None
    

def create_graph (history, graph_path, filename):
    plt.title(f'Training Loss - {history["mean"]:.2f} +- {history["variance"]**0.5:.2f}')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.ylabel('Loss')
    for i, train in enumerate(history['training']):
        epochs = range(len(train))
        plt.plot(epochs, train, linestyle='-', label=f'Training_{i}_fold loss')
    for i, val in enumerate(history['validation']):
        epochs = [x*history['val_step'] for x in range(len(val))]
        plt.plot(epochs, val, linestyle='--', label=f'Validation_{i}_fold loss')
    # for i, wc in enumerate(history['weight_changes']):
    #     epochs = [x*history['val_step'] for x in range(len(val))]
    #     plt.plot(epochs, wc, linestyle='-.', label=f'WC_{i}_fold loss')
    # #val_path = os.path.join(graph_path, 'validation')
    # train_path = os.path.join(graph_path, 'training')
    # if (not os.path.exists(train_path)):
    #     os.makedirs(train_path)
    plt.legend()
    plt.savefig(os.path.join(graph_path, filename))
    plt.clf()

    # plt.title(f'Maximum of Gradients - {history["mean"]:.2f} +- {history["variance"]**0.5:.2f}')
    # plt.xlabel('Epochs')
    # #plt.yscale('log')
    # plt.ylabel('Values')
    # colors = ['c', 'm', 'y', 'k', 'c', 'm', 'y', 'k']
    # lines = ['-', '-.', '--', ':']
    # for i, gradients in enumerate (history['gradients']):
    #     for layer, gradient in enumerate(gradients):
    #         epochs = range(len(gradient))
    #         plt.plot(epochs, gradient, colors[i], linestyle=lines[layer], label=f'{layer}th layer max gradient of {i}_fold')
    # grad_path = os.path.join(graph_path, 'gradients')
    # if (not os.path.exists(grad_path)):
    #     os.makedirs(grad_path)
    # plt.legend()
    # plt.savefig(os.path.join(grad_path, filename))
    # plt.clf()

    # plt.title(f'Validation Loss - AVG +- VAR')
    # plt.xlabel('Epochs')
    # plt.yscale('log')
    # plt.ylabel('Loss')
    
    # if (not os.path.exists(val_path)):
    #     os.makedirs(val_path)
    # plt.legend()
    # plt.savefig(os.path.join(val_path, filename))
    # plt.clf()

def count(dl, global_confs, local_confs, output_path, graph_path, seed=4444):
    history = {}
    history['mean'] = 1.
    layers      = local_confs["layers"]
    batch_size  = local_confs["batch_size"]
    eta_decay   = local_confs["eta_decay"]#if == -1 no eta decay; 25 should be fine
    eta         = local_confs["eta"]
    lam         = local_confs["lambda"]
    alpha       = local_confs["alpha"] 
    history['hyperparameters'] = (layers, batch_size, eta, lam, alpha, eta_decay)
    return history

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
    seed = int(config.get("seed", args.seed)) #prendiamo dal file di config, e se non c'è prendiamo da riga di comando. il default è 2021
    print(f"seed: {seed}")
    set_seed(seed)

    ### loading and preprocessing dataset from config ###
    dl = DataLoader(seed)
    dl.load_data(config["train_set"], config["input_size"], config["output_size"], config.get("preprocessing"))

    ### loading CONSTANT parameters from config ###
    global_conf = config["model"]["global_conf"]

    ### loding hyperparameters from config ###
    hyperparameters = config["model"]["hyperparameters"]
    #each configuration is a triple: datas, global confs and local confs
    configurations = [
        (dl, global_conf, 
         {"layers": layers,
          "batch_size": batch_size, 
          "eta": eta,
          "lambda": lam, 
          "alpha": alpha,
          "eta_decay": eta_decay},
         output_path,
         graph_path,
         seed
        )
        for layers      in hyperparameters["hidden_units"]
        for batch_size  in hyperparameters["batch_size"]
        for eta         in hyperparameters["eta"]
        for lam         in hyperparameters["lambda"]
        for alpha       in hyperparameters["alpha"]
        for eta_decay   in hyperparameters["eta_decay"]
    ]

    ### training ###
    if (not args.nested):
        with Pool() as pool:
            try:
                results = pool.starmap(train, configurations)
                print(f"numero iterazioni: {len(results)}")
            except KeyboardInterrupt:
                pool.terminate()
                print("forced termination")
                exit()
    else:
        results = []
        shrink = args.shrink
        loop = args.loop
        for i in range(loop):
            with Pool() as pool:
                try:
                    result_it = pool.starmap(train, configurations)
                except KeyboardInterrupt:
                    pool.terminate()
                    print("forced termination")
                    exit()
            results.extend (result_it)
            print(f"numero iterazioni: {len(result_it)}")
            
            test_vs_hyper = { i : history['mean'] for i, history in enumerate(results) }
            best3 = heapq.nsmallest(3, test_vs_hyper)
            print(best3)
            best_hyper = [ results[best]['hyperparameters'] for best in best3 ]
            configurations = []
            for layers, batch_size, eta, lam, alpha, eta_decay in best_hyper:
                eta_new = []
                lam_new = []
                alpha_new = []
                eta_new.append(eta)
                eta_new.append(eta + (shrink))
                eta_new.append(eta - (shrink))
                lam_new.append(lam)
                if (lam != 0):
                    lam_new.append(10**(np.log10(lam) + 3*(shrink))) #1e-4 --> 1e-3.9 e 1e-4.1
                    lam_new.append(10**(np.log10(lam) - 3*(shrink)))
                alpha_new.append(alpha)
                alpha_new.append(alpha + (shrink))
                alpha_new.append(alpha - (shrink))

                configurations.extend( [
                    (dl, global_conf, 
                    {"layers": layers,
                    "batch_size": batch_size, 
                    "eta": eta,
                    "lambda": lam, 
                    "alpha": alpha,
                    "eta_decay": eta_decay},
                    output_path,
                    graph_path,
                    seed
                    )
                    for eta         in list(set(eta_new))
                    for lam         in list(set(lam_new))
                    for alpha       in list(set(alpha_new))
                ])
            shrink *= shrink
            print("a cycle of nest has ended")

        
    ##here goes model selectiom
    print("grid search complete!")



if __name__ == '__main__':
    main()
