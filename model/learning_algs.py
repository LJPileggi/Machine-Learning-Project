import heapq
from copy import deepcopy
import itertools
import os
from types import SimpleNamespace

from matplotlib.pyplot import hist
import joblib
from multiprocessing import Pool
import numpy as np
from numpy.core.numeric import empty_like
from MLP import MLP
# from configuration import Configuration
from dataloader import DataLoader
from history import History, Results

def train(TR, VL, TS, global_confs, hyp):
    hyp = SimpleNamespace(**hyp)
    history = History(global_confs.metrics)
    input_size = DataLoader.get_input_size_static(TR) 
    #initializing MLP and history
    nn = MLP (global_confs.seed, global_confs.task, input_size, hyp.layers)
    history.update_plots(nn, train=TR, val=VL, test=TS)
    
    #training loop
    oldWeights = nn.get_weights()
    low_wc = 0
    for epoch in range (global_confs.max_step):
        
        for current_batch in DataLoader.dataset_partition_static(TR, hyp.batch_size):
            for pattern in current_batch:
                out = nn.forward(pattern[0])
                error = pattern[1] - out
                nn.backwards(error)
                #we are updating with eta/TS_size in order to compute LMS, not simply LS
            len_batch = len(TR) #if batch_size != 1 else len(whole_TR)
            if hyp.eta_decay == -1:
                nn.update_weights(hyp.eta/len_batch, hyp.lam, hyp.alpha)
            else:
                nn.update_weights((0.9*np.exp(-(epoch)/hyp.eta_decay)+0.1)*hyp.eta/len_batch, hyp.lam, hyp.alpha)
        
        #after each epoch
        history.update_plots(nn, train=TR, val=VL, test=TS)
        if(epoch % global_confs.check_step == 0):
            #once each check_step epoch
            #print validation error
            
            metric = global_confs.gs_preferred_metric
            err = history.get_last_error(*metric) 
            print(f"{epoch} - banana - {metric}: {err}")
            #compute weights change
            newWeights = nn.get_weights()
            wc = np.mean(np.abs((oldWeights-newWeights)/oldWeights))
            oldWeights = newWeights
            
            #stopping criteria
            if wc <= global_confs.wc_threshold:
                low_wc +=1
            else:
                low_wc = 0
            desired_value = 1 if metric[1] == "accuracy" else 0
            if (np.allclose(err, desired_value, atol=global_confs.epsilon)):
                print(f"endend in {epoch} epochs!")
                break
            if (low_wc >= global_confs.patience):
                break
    #once training has ended
    return history, nn #cosa restituisce davvero?


def cross_val(TR, TS, global_confs, hyp, output_path, graph_path):
    try:
        #fare un for max_fold, e per ogni fold, recuperare il whole_TR, whole_VR ecc. Poi si prendono le medie del testing e si printano i grafici di tutti.
        # results = {set: 
        #             {metric: 
        #                     {"mean": 0, "variance": 0} 
        #              for metric in global_confs["metrics"]}
        #            for set in global_confs["datasets"]}
        results = Results(hyp, global_confs.metrics)
        for n_fold, (TR, VL) in enumerate (DataLoader.get_slices_static(TR, global_confs.maxfold)):
            history, nn = train(TR, VL, TS, global_confs, hyp)
            results.add_history(history)
            #print(f"accuracy - {history['name']}: {(history['testing'][n_fold])}")
            ### saving model ###
            filename =  f"model_{results.name}_{n_fold}fold.logami"
            path = os.path.join(output_path, filename)
            joblib.dump (nn, path)
        ### plotting loss###
        results.calculate_mean()
        results.create_graph(graph_path)
        return results
    except KeyboardInterrupt:
        print('Interrupted')
        return None

# def get_children_paremetrs(hyper, shrink, parameters):    
#     eta_new = [
#         hyper['eta'], hyper['eta']+(shrink), hyper['eta']-(shrink)
#         ]

#     lam_new = [hyper['lam']]
#     if (hyper['lam'] != 0):
#         lam_new.append(10**(np.log10(hyper['lam']) + 3*(shrink))) #1e-4 --> 1e-3.9 e 1e-4.1
#         lam_new.append(10**(np.log10(hyper['lam']) - 3*(shrink)))
    
#     alpha_new = [
#         hyper['alpha'], hyper['alpha']+(shrink), hyper['alpha']-(shrink)
#         ]

#     def my_product(inp):
#         return (dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values()))
    

#     return [
#             {"layers": hyper['layers'],
#              "batch_size": hyper['batch_size'], 
#              "eta_decay": hyper['eta_decay'],
#              "eta": eta,
#              "lambda": lam, 
#              "alpha": alpha
#             }
#         for eta, lam, alpha, layers, batchsize, etadecay, in configs
#     ]


def copy_and_update(old_dict, new_dict):
    d = old_dict
    out = deepcopy(d)
    out.update(new_dict)
    return out

def get_children(hyper, searched_hyper, shrink):
    new_hyper = {}
    for key, value in hyper.items():
        if key in searched_hyper:
            if key == 'lam':
                if value != 0:
                    new_hyper.update({key:[10**(np.log10(value) - 3*(shrink)), value, 10**(np.log10(value) + 3*(shrink))]})
                else:
                    new_hyper.update({key:[value]})
            else:
                new_hyper.update({key:[value-shrink, value, value+shrink]})
        else:
            new_hyper.update({key:[value]})

    new_configs = [dict(zip(new_hyper.keys(), values)) for values in itertools.product(*new_hyper.values())]
    return [
        copy_and_update(hyper, new_config)
        for new_config in new_configs
    ]

def grid_search(TR, TS, global_conf, hyper, output_path, graph_path, loop=1, shrink=0.1):
    results = []

    # searched_hyper = []
    # for key, value in hyper.items():
    #     if (key != "hidden_units") or (key != "batch_size") or (key != "eta_decay"):
    #         searched_hyper.append(key)
    # print(searched_hyper)

    
    # configurations = generate configurations
    # train a model for each configuration
    # new configurations = flatten(get_children(model) for model in best model)
    # train a model for each configuration


    #configuration used for the first grid search
    configurations = [ 
        {"layers": layers, 
         "batch_size": batch_size,
         "eta_decay": eta_decay,
         "eta": eta,
         "lam": lam,
         "alpha": alpha}
        
        for layers      in hyper["hidden_units"]
        for batch_size  in hyper["batch_size"]
        for eta         in hyper["eta"]
        for lam         in hyper["lam"]
        for alpha       in hyper["alpha"]
        for eta_decay   in hyper["eta_decay"]
    ]

    selected_metric = tuple(global_conf.gs_preferred_metric)
    for i in range(loop): #possibly just one iteration
        #training the configs of the previous step
        print("starting a grid search cycle")
        with Pool() as pool:
            try:
                async_results = [
                    pool.apply_async(cross_val, (TR, TS, global_conf, hyp, output_path, graph_path)) 
                    for hyp in configurations
                ]
                pool.close()
                pool.join()
                # result_it = pool.starmap(train, configurations)
                result_it = list(map(lambda async_result: async_result.get(), async_results))
            except KeyboardInterrupt:
                print("forcing termination")
                pool.terminate()
                #pool.join()
                print("forced termination")
                exit()
        results.extend (result_it)
        print("a cycle of nest has ended")
        print(f"models trained: {len(result_it)}")

        #building the config of the next step
        configurations = []
        #we should order w.r.t. which metric? on which set?
        results.sort(key=lambda result: result.results[selected_metric]['mean'])
        best_hyper = [ best.hyperparameters for best in results[:3] ]
        print(f"i migliori 3 modelli di sto ciclo sono: {best_hyper}")
        configurations = []
        for best in best_hyper:
            configurations.extend(get_children(best, global_conf.searched_hyper, shrink))
        shrink *= shrink
        
        
    results.sort(key=lambda result: result.results[selected_metric]['mean'])
    best_hyper = [ best.hyperparameters for best in results[:3] ]
    print(f"i miglior modelli di questa nested sono: {best_hyper}")

    return best_hyper[0]
