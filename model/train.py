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
def train(dl, confs, layers, batch_size, eta, lam, alpha):
    try:
        print(f"\neta: {eta}\nlambda: {lam}\nbatch_size={batch_size}\nalpha: {alpha}")
        
        #set global configurations#
        activation = confs["activation_units"]
        max_step   = confs["max_step"]
        check_step = confs["check_step"]
        epsilon    = confs["epsilon"]
        input_size = dl.get_input_size ()
        whole_TR = dl.get_partition_set('train')
        whole_VL = dl.get_partition_set('val')
        
        ##setting history to store plots data
        history = {}
        history['training'] = []
        history['validation'] = []
        history['val_step'] = check_step
        history['name'] = f"{layers}_{batch_size}_{eta}_{lam}_{alpha}"

        #create mlp#
        nn = MLP (input_size, layers, activation)
        
        #whatch out! if batch_size = -1, it becomes len(TR)
        batch_size = len(whole_TR) if batch_size == -1 else batch_size
        
        #prepares variables used in epochs#
        train_err = np.inf
        val_err = np.inf
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
            if(i % check_step == 0):
                #once each check_step epoch
                val_err = MSE_over_network (whole_VL, nn)
                history['validation'].append(train_err)
                print (f"{i} - {history['name']}: {train_err} - {val_err}")
                if (np.allclose(val_err, 0, atol=epsilon)):
                    break
        history['testing'] = accuracy (dl.get_partition_set('test'), nn) * 100
        print(f"accuracy - {history['name']}: {(history['testing'])}%") 
        return history, nn
    except KeyboardInterrupt:
        print('Interrupted')
        return None
    

def create_graph (history, filename):
    epochs = range(len(history['training']))
    val_epochs = [x*history['val_step'] for x in range(len(history['validation']))]
    plt.plot(epochs, history['training'], 'b', label=f'Training_{history["name"]} loss')
    plt.plot(val_epochs, history['validation'], 'g', label=f'Validation_{history["name"]} loss')
    print(f"{history['testing'][0]:.2f}")
    plt.title(f'Training and Validation Loss - {history["testing"][0]:.2f}')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.clf()

if __name__ == '__main__':
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
    configurations = [
        (dl, global_conf, layers, batch_size, eta, lam, alpha)
        for layers      in hyperparameters["hidden_units"]
        for batch_size  in hyperparameters["batch_size"]
        for eta         in hyperparameters["eta"]
        for lam         in hyperparameters["lambda"]
        for alpha       in hyperparameters["alpha"]
    ]

    ### training ###
    with Pool() as pool:
        results = pool.starmap(train, configurations)
    # for (layers, batch_size, eta, lam, alpha) in hyperparameters:
    #     #it reads as nonlocal variables: dl, activation, checkstep, maxstep, epsilon    
    #     print(f"\neta: {eta}\nlambda: {lam}\nbatch_size={batch_size}\nalpha: {alpha}")
    #     train_err, val_err, nn = train(layers, batch_size, eta, lam, alpha)

    #     ### printing results ###
    #     print(f"train_err: {np.array(train_err)}")
    #     test_error = accuracy (dl.get_partition_set('test'), nn)
    #     print(f"accuracy: {(test_error)*100}%") 

    ### plotting loss ###
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
    for history, nn in results:
        graph_name = f"training_loss_{history['name']}.png"
        create_graph(history, os.path.join(graph_path, graph_name))
        ### saving model ###
        nn.save_model(os.path.join(output_path, f"model_{history['name']}.h5"))

