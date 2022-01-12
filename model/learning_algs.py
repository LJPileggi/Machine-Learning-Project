import heapq
import itertools
import os

from matplotlib.pyplot import hist
import joblib
from multiprocessing import Pool
import numpy as np
from numpy.core.numeric import empty_like
from MLP import MLP
from model.configuration import Configuration
from model.history import History





def train(seed, input_size, TR, VL, TS, global_confs, hyp):
    #initializing MLP and history
    nn = MLP (global_confs["task"], input_size, hyp.layers, seed)
    history = History(global_confs["datasets"], global_confs["metrics"], hyp)
    history.update_plots(nn, tr=TR, vl=VL, ts=TS)
    
    #training loop
    oldWeights = nn.get_weights()
    low_wc = 0
    for epoch in range (global_confs["max_step"]):
        
        for current_batch in dl.dataset_partition(TR, hyp.batch_size):
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
        history.update_plots(nn, tr=TR, vl=VL, ts=TS)
        if(epoch % global_confs["check_step"] == 0):
            #once each check_step epoch
            #print validation error
            val_err = history.plots["val"]["mee"][-1]
            print(f"{epoch} - {history['name']} - val err: {val_err}")
            
            #compute weights change
            newWeights = nn.get_weights()
            wc = np.mean(np.abs((oldWeights-newWeights)/oldWeights))
            oldWeights = newWeights
            
            #stopping criteria
            if wc <= global_confs["threshold"]:
                low_wc +=1
            else:
                low_wc = 0 
            if (np.allclose(val_err, 0, atol=global_confs["epsilon"]) or low_wc >= global_confs["patience"]):
                break
    #once training has ended
    return history, nn


def cross_val(dl, seed, global_confs, hyp, output_path, graph_path):
    try:
        #fare un for max_fold, e per ogni fold, recuperare il whole_TR, whole_VR ecc. Poi si prendono le medie del testing e si printano i grafici di tutti.
        input_size    = dl.get_input_size ()
        results = {set: 
                    {metric: 
                            {"mean": 0, "variance": 0} 
                     for metric in global_confs["metrics"]}
                   for set in global_confs["datasets"]}
        TS = dl.getTS()
        for n_fold, TR, VL in enumerate (dl.get_slices(global_confs["max_fold"])):
            nn, history = train(seed, input_size, TR, VL, TS, global_confs, hyp)
            for set in global_confs["sets"]:
                for metric in global_confs["metrics"]:
                    final_error = history[set][metric][-1]
                    results[set][metric]["mean"] += final_error/global_confs["max_fold"]
                    results[set][metric]["variance"] += (final_error**2)/global_confs["max_fold"]
            #print(f"accuracy - {history['name']}: {(history['testing'][n_fold])}")
            ### saving model and plotting loss ###
            filename =  f"model_{history['name']}_{n_fold}fold.logami"
            path = os.path.join(output_path, filename)
            joblib.dump (nn, path)
        for set in global_confs["sets"]:
                for metric in global_confs["metrics"]:
                    results[set][metric]['variance'] -= results[set][metric]['mean']**2
        ### plotting loss ###
        ds.create_graph(history, f"training_loss_{history['name']}.png", graph_path)
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


def get_children(hyper, searched_hyper, shrink):
    new_hyper = {}
    for key, value in vars(hyper).items():
        if key in searched_hyper:
            if key == 'lam':
                if value != 0:
                    new_hyper.update({key:[10**(np.log10(value) - 3*(shrink), value, 10**(np.log10(value) + 3*(shrink))]})
                else:
                    new_hyper.update({key:[value]})
            else:
                new_hyper.update({key:[value-shrink, value, value+shrink]})
        else:
            new_hyper.update({key:[value]})

    new_configs = Configuration.my_product(new_hyper)
    return [
        hyper.get_copy_with(new)
        for new in new_configs
    ]




def grid_search(seed, dl, ds, global_conf, hyper, output_path, graph_path, loop=1, shrink=0.1):
    results = []
    searched_hyper = []
    for key, value in hyper.items():
        if (key != "hidden_units") | (key != "batch_size"):
            if len(value) > 1:
                searched_hyper.append(key)
    
    # configurations = generate configurations
    # train a model for each configuration
    # new configurations = flatten(get_children(model) for model in best model)
    # train a model for each configuration


    #configuration used for the first grid search
    configurations = [ 
        Configuration(layers, batch_size, eta_decay, eta, lam, alpha)
        
        for layers      in hyper["hidden_units"]
        for batch_size  in hyper["batch_size"]
        for eta         in hyper["eta"]
        for lam         in hyper["lambda"]
        for alpha       in hyper["alpha"]
        for eta_decay   in hyper["eta_decay"]
    ]

    for i in range(loop): #possibly just one iteration
        #training the configs of the previous step
        with Pool() as pool:
            try:
                async_results = [
                    pool.apply_async(cross_val, dl, seed, global_conf, config, output_path, graph_path)
                    for config in configurations
                ]
                pool.close
                pool.join()
                # result_it = pool.starmap(train, configurations)
                result_it = map(lambda async_result: async_result.get(), async_results)
            except KeyboardInterrupt:
                print("forcing termination")
                pool.terminate()
                pool.join()
                print("forced termination")
                exit()
        results.extend (result_it)
        print("a cycle of nest has ended")
        print(f"models trained: {len(result_it)}")

        #building the config of the next step
        configurations = []
        #we should order w.r.t. wich metric? on which set?
        results.sort(key=lambda result: result['mean'])
        best_hyper = [ best['hyperparameters'] for best in results[:3] ]
        print(f"i migliori 3 modelli di sto ciclio sono: {best_hyper}")
        configurations = []
        for best in best_hyper:
            configurations.append(get_children(best, searched_hyper, shrink))
        shrink *= shrink
        
        
    
    test_vs_hyper = { i : history['mean'] for i, history in enumerate(results) }
    best3 = heapq.nsmallest(1, test_vs_hyper)
    best_hyper = [ results[best]['hyperparameters'] for best in best3 ]
    print(f"il migliore modello di sta nested è: {best_hyper}")
