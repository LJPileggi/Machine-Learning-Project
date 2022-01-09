import heapq
from multiprocessing import Pool
from matplotlib.pyplot import hist
import numpy as np

def empirical_error(batch, NN, metric):
    error = 0
    if metric == "missclass":
        for pattern in batch:
            error += np.max(abs(NN.h(pattern[0]) - pattern[1]))
        return error/len(batch)
    if metric == "accuracy":
        for pattern in batch:
            error += np.max(abs(NN.h(pattern[0]) - pattern[1]))
        return 1-error/len(batch)
    if metric == "mse":
        for pattern in batch:
            error += ((NN.forward(pattern[0]) - pattern[1])**2).sum() 
        return error/len(batch)
    elif metric == "mee":
        for pattern in batch:
            error += ((NN.forward(pattern[0]) - pattern[1])**2).sum()**1/2
        return error/len(batch)
    else:
        raise NotImplementedError("unknown metric")




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
        history["hyperparameters"] = local_confs   
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



def grid_search(seed, dl, global_conf, hyperparameters, loop=1, shrink=0.1):
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
    results = []
    for i in range(loop): #possibly just one iteration
        with Pool() as pool:
            try:
                result_it = pool.starmap(train, configurations)
            except KeyboardInterrupt:
                pool.terminate()
                print("forced termination")
                exit()
        results.extend (result_it)
        print(f"models trained: {len(result_it)}")
        
        test_vs_hyper = { i : history['mean'] for i, history in enumerate(results) }
        best3 = heapq.nsmallest(3, test_vs_hyper)
        best_hyper = [ results[best]['hyperparameters'] for best in best3 ]
        print(f"i migliori 3 modelli di sto ciclio sono: {best_hyper}")
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
                for eta         in eta_new
                for lam         in lam_new
                for alpha       in alpha_new
            ])
        shrink *= shrink
        print("a cycle of nest has ended")
    
    test_vs_hyper = { i : history['mean'] for i, history in enumerate(results) }
    best3 = heapq.nsmallest(1, test_vs_hyper)
    best_hyper = [ results[best]['hyperparameters'] for best in best3 ]
    print(f"il migliore modello di sta nested è: {best_hyper}")

def cross_val():
    pass