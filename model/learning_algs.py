import heapq
from copy import deepcopy
import itertools
import os
import time
from types import SimpleNamespace
import joblib
from multiprocessing import Pool
import numpy as np
from numpy.core.numeric import empty_like
from MLP import MLP
# from configuration import Configuration
from dataloader import DataLoader
from history import History, Results

def train(TR, VL, TS, global_confs, hyp, preproc):
    """Train function
    The main function used to train a model
    Args:
        TR (ndarray): The training set
        VL (ndarrya): The validation set, it can be none
        TS (ndarray): The test set, it can be none
        global_confs (Singlenamespace): the global config, fixed for all of the models we are going to generate
        hyp (Configuration): the hyperparameters that define the model we are going to train
        preproc (ndarray): an array detailing how we are going to preprocess the input or output

    Returns:
        history (History): an history object with all the value of TR, VL, TS over the epochs of training
        nn (MLP): the trained neural network
    """
    hyp = SimpleNamespace(**hyp)
    history = History(global_confs.metrics)
    input_size = DataLoader.get_input_size_static(TR) 
    #initializing MLP and history
    nn = MLP (global_confs.seed, global_confs.task, input_size, hyp.layers, preproc)
    
    #scaling inputs
    scaledTR = nn.scale_dataset(TR)
    
    #training loop
    oldWeights = nn.get_weights()
    low_loss = 0
    low_wc = 0
    for epoch in range (1, global_confs.max_step+1):
        for current_batch in DataLoader.dataset_partition_static(scaledTR, hyp.batch_size):
            patterns, labels = list(map(np.array, list(zip(*current_batch))))
            outs = nn.forward(patterns)
            errors = labels - outs
            nn.backwards(errors)
            
            #we are updating with eta/TS_size in order to compute LMS, not simply LS
            len_batch = len(scaledTR) #if batch_size != 1 else len(whole_TR)
            if hyp.eta_decay == -1:
                nn.update_weights(hyp.eta/len_batch, hyp.lam, hyp.alpha)
            else:
                nn.update_weights((0.9*np.exp(-(epoch)/hyp.eta_decay)+0.1)*hyp.eta/len_batch, hyp.lam, hyp.alpha)
        
        #after each epoch
        history.update_plots(nn, train=TR, val=VL, test=TS)
        if(epoch % global_confs.check_step == 0):
            #once each check_step epoch
            #print validation error
            
            #retrieve preferred metric
            metric = global_confs.gs_preferred_metric
            err = history.get_last_error(*metric) 
            #compute weights change
            newWeights = nn.get_weights()
            wc = np.mean(np.abs((oldWeights-newWeights)/(oldWeights+0.001)))
            oldWeights = newWeights
            #print both
            print(f"{epoch} - {metric}: {err} - wc: {wc}")
               
            #preferred metric stopping criterion
            desired_value = 1 if metric[1] == "accuracy" else 0
            if (np.allclose(err, desired_value, atol=global_confs.loss_tolerance)):
                low_loss += 1
            else:
                low_loss = 0
            if (low_loss >= global_confs.loss_patience):
                print("loss convergence reached")
                print(f"endend in {epoch} epochs!")
                break
            #weight change stopping criterion
            if wc <= global_confs.wc_tolerance:
                low_wc +=1
            else:
                low_wc = 0
            if (low_wc >= global_confs.wc_patience):
                print("wc convergence reached")
                print(f"endend in {epoch} epochs!")
                break
            
    #once training has ended
    return history, nn #cosa restituisce davvero?


def cross_val(TR, TS, global_confs, hyp, output_path, graph_path, preproc):
    try:
        results = Results(hyp, global_confs.metrics)
        for n_fold, (TR, VL) in enumerate (DataLoader.get_slices_static(TR, global_confs.maxfold)):
            history, nn = train(TR, VL, TS, global_confs, hyp, preproc)
            results.add_history(history)

            ### saving model ###
            filename =  f"model_{results.name}_{n_fold}fold.logami"
            path = os.path.join(output_path, filename)
            joblib.dump (nn, path)
        ### plotting loss###
        results.calculate_mean(graph_path)
        results.create_graph(graph_path)
        return results
    except KeyboardInterrupt:
        print('Interrupted')
        return None

def multiple_trials(TR, TS, global_confs, hyp, output_path, graph_path, preproc):
    try:
        print("started multiple trials")
        results = Results(hyp, global_confs.metrics)
        for n_trial in range(global_confs.maxfold):
            print(f"trial nr. {n_trial}")
            TR, VL = TR, None
            global_confs.seed += n_trial*1729
            history, nn = train(TR, VL, TS, global_confs, hyp, preproc)
            results.add_history(history)
            ### saving model ###
            filename =  f"model_{results.name}_{n_trial}trial.logami"
            path = os.path.join(output_path, filename)
            joblib.dump (nn, path)
        ### plotting loss###
        results.calculate_mean(graph_path)
        results.create_graph(graph_path)
        return results
    except KeyboardInterrupt:
        print('Interrupted')
        return None

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
                scaled_shrink = shrink*(10**(np.floor(np.log10(value))))
                new_hyper.update({key:[value-scaled_shrink, value, value+scaled_shrink]})
        else:
            new_hyper.update({key:[value]})

    new_configs = [dict(zip(new_hyper.keys(), values)) for values in itertools.product(*new_hyper.values())]
    return [
        copy_and_update(hyper, new_config)
        for new_config in new_configs
    ]

def grid_search(TR, TS, global_conf, hyper, output_path, graph_path, preproc, loop=1, shrink=0.1):
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
    for i in range(int(loop)): #possibly just one iteration
        #training the configs of the previous step
        print("starting a grid search cycle")
        pool = Pool()
        f = multiple_trials if global_conf.validation == "trials" else cross_val
        async_results = [
            pool.apply_async(f, (TR, TS, global_conf, hyp, output_path, graph_path, preproc)) 
            for hyp in configurations
        ]
        pool.close()
        pool.join()
        # result_it = pool.starmap(train, configurations)
        result_it = list(map(lambda async_result: async_result.get(), async_results))
        results.extend (result_it)
        print("a cycle of nest has ended")
        print(f"models trained: {len(result_it)}")

        #building the config of the next step
        configurations = []
        #we should order w.r.t. which metric? on which set?
        results = list(filter(lambda r: r.results[selected_metric]['variance'] <= global_conf.gs_max_variance, results))
        results.sort(key=lambda result: result.results[selected_metric]['mean'])
        best_hyper = [ best.hyperparameters for best in results[:3] ]
        print(f"i migliori 3 modelli di sto ciclo sono: {best_hyper}")
        configurations = []
        for best in best_hyper:
            configurations.extend(get_children(best, global_conf.searched_hyper, shrink))
        shrink *= 0.1
        
        
    results = list(filter(lambda r: r.results[selected_metric]['variance'] <= global_conf.gs_max_variance, results))
    results.sort(key=lambda result: result.results[selected_metric]['mean'])
    best_hyper = [ best.hyperparameters for best in results[:3] ]
    print(f"i miglior modelli di questa nested sono: {best_hyper}")
    fname = os.path.join(graph_path, "best_reults.txt")
    with open(fname, "w") as f:
        f.write(f"i miglior modelli di questa nested sono: {best_hyper}")


    return best_hyper[0]
