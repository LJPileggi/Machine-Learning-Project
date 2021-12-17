from dataloader import DataLoader
from MLP import MLP
from multiprocessing import Pool
from datetime import datetime
import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt



def MSE_over_network(batch, NN):
    mse = 0
    for pattern in batch:
        out = NN.forward(pattern[0])
        out = out > 0.5
        mse += ((out - pattern[1])**2)
    mse = mse/len(batch)
    return mse

def accuracy (batch, NN):
    errors = 0
    for pattern in batch:
        out = NN.forward(pattern[0])
        out = out > 0.5
        errors += (abs(out - pattern[1]))
    accuracy = 1 - errors/len(batch)
    return accuracy

#it reads as nonlocal variables: dl, activation, checkstep, maxstep, epsilon
def train(dl, global_confs, local_confs, output_path, graph_path):
    try:
        #accessing data
        input_size = dl.get_input_size ()
        whole_TR = dl.get_partition_set('train')
        whole_VL = dl.get_partition_set('val')
        
        #set global configurations#
        activation  = global_confs["activation_units"]
        max_step    = global_confs["max_step"]
        check_step  = global_confs["check_step"]
        epsilon     = global_confs["epsilon"]

        #set local configuration
        layers      = local_confs["layers"]
        batch_size  = local_confs["batch_size"]
        eta         = local_confs["eta"]
        lam         = local_confs["lambda"]
        alpha       = local_confs["alpha"]
        patience    = local_confs["patience"]

        #create mlp#
        nn = MLP (input_size, layers, activation)
        
        #setting history to store plots data
        history = {}
        history['training'] = []
        history['validation'] = []
        history['gradients'] = [ [] for layer in layers]
        history['val_step'] = check_step
        history['name'] = f"{layers}_{batch_size}_{eta}_{lam}_{alpha}"

        
        
        #prepares variables used in epochs#
        train_err = np.inf
        val_err = np.inf
        old_val_err = np.inf
        val_err_plateau = 1 #a "size 1 plateau" is just one point
        #whatch out! if batch_size = -1, it becomes len(TR)
        batch_size = len(whole_TR) if batch_size == -1 else batch_size
        for i in range (max_step):
            for current_batch in dl.dataset_partition('train', batch_size):
                for pattern in current_batch:
                    out = nn.forward(pattern[0])
                    error = pattern[1] - out
                    nn.backwards(error)
                #we are updating with eta/TS_size in order to compute LMS, not simply LS
                nn.update_weights(eta/len(whole_TR), lam, alpha)
            #after each epoch
            train_err = MSE_over_network (whole_TR, nn)
            history['training'].append(train_err)
            for layer, grad in enumerate(nn.get_max_grad_list()):
                history['gradients'][layer].append(grad)
            if(i % check_step == 0):
                #once each check_step epoch
                val_err = MSE_over_network (whole_VL, nn)
                history['validation'].append(train_err)
                print (f"{i} - {history['name']}: {train_err} - {val_err}")
                if val_err == old_val_err:
                    val_err_plateau += 1
                else:
                    val_err_plateau = 1
                if (np.allclose(val_err, 0, atol=epsilon) and val_err_plateau >= patience):
                    break
                old_val_err = val_err
        history['testing'] = accuracy (dl.get_partition_set('test'), nn) * 100
        print(f"accuracy - {history['name']}: {(history['testing'])}%") 

        ### saving model and plotting loss ###
        create_graph(history, os.path.join(graph_path, f"training_loss_{history['name']}.png"))
        nn.save_model(os.path.join(output_path, f"model_{history['name']}.h5"))



        return history, nn
    except KeyboardInterrupt:
        print('Interrupted')
        return None
    

def create_graph (history, filename):
    epochs = range(len(history['training']))
    val_epochs = [x*history['val_step'] for x in range(len(history['validation']))]
    plt.plot(epochs, history['training'], 'b', label=f'Training_{history["name"]} loss')
    colors = ['c', 'm', 'y', 'k', 'c', 'm', 'y', 'k']
    for layer, gradient in enumerate(history['gradients']):
        plt.plot(epochs, gradient, colors[layer], label=f'{layer}th layer max gradient')
    plt.plot(val_epochs, history['validation'], 'g', label=f'Validation_{history["name"]} loss')
    print(f"{history['testing'][0]:.2f}")
    plt.title(f'Training and Validation Loss - {history["testing"][0]:.2f}')
    plt.xlabel('Epochs')
    #plt.yscale('log')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def main():
    ### Parsing cli arguments ###
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument('--config_path',
                        help='path to config file')
    parser.add_argument('--graph_name',
                        help='name of the loss graph you will generate')
    parser.add_argument('--grid_search', dest='grid_search', action='store_true',
                        help='If you are going to do a grid_search')
    parser.set_defaults(grid_search=False)
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    ### setting up output directories ###
    now = datetime.now()
    date = str(datetime.date(now))
    time = str(datetime.time(now))
    time = time[:2] + time[3:5]
    output_path = os.path.abspath(config["output_path"])
    output_path = os.path.join(output_path, date, time)
    print(output_path)
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    graph_path = os.path.abspath(config["graph_path"])
    graph_path = os.path.join(graph_path, date, time)
    print(graph_path)
    if (not os.path.exists(graph_path)):
        os.makedirs(graph_path)

    ### loading and preprocessing dataset from config ###
    train_set  = config["train_set"]
    test_set   = config["test_set"]
    encoding   = config["preprocessing"]["1_hot_enc"]
    dl = DataLoader ()
    dl.load_data_from_dataset(test_set, encoding, train_slice=0.5)
    


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
          "patience": patience},
          output_path,
          graph_path
        )
        for layers      in hyperparameters["hidden_units"]
        for batch_size  in hyperparameters["batch_size"]
        for eta         in hyperparameters["eta"]
        for lam         in hyperparameters["lambda"]
        for alpha       in hyperparameters["alpha"]
        for patience    in hyperparameters["patience"]
    ]

    ### training ###
    with Pool() as pool:
        try:
            results = pool.starmap(train, configurations)
        except KeyboardInterrupt:
           pool.terminate()
           print("forced termination")
           exit()
    
    ##here goes model selectiom
    print("training complete!")



if __name__ == '__main__':
    main()
