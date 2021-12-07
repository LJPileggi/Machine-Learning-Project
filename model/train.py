from dataloader import DataLoader
from MLP import MLP

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
def train(layers, batch_size, eta, lam, alpha):
    input_size = dl.get_input_size ()
    whole_TR = dl.get_partition_set('train')
    #whatch out! if batch_size = -1, it becomes len(TR)
    batch_size = len(whole_TR) if batch_size == -1 else batch_size
    whole_VL = dl.get_partition_set('val')
    nn = MLP (input_size, layers, activation)
    train_err = np.inf
    val_err = np.inf
    
    all_train_err = []
    all_val_err = []
    for i in range (max_step):
        for current_batch in dl.dataset_partition('train', batch_size):
            for pattern in current_batch:
                out = nn.forward(pattern[0])
                error = pattern[1] - out
                nn.backwards(error)
            #we are updating with eta/TS_size in order to compute LMS, not simply LS
            nn.update_weights(eta/len(whole_TR), lam, alpha)
        train_err = MSE_over_network (whole_TR, nn)
        all_train_err.append(train_err)
        if(i % check_step == 0):
            val_err = MSE_over_network (whole_VL, nn)
            print (f"{i}: {train_err} - {val_err}")
            all_val_err.append(val_err)
            if (np.allclose(val_err, 0, atol=epsilon)):
                break
    return all_train_err, all_val_err, nn

def create_graph (train_err, val_err, filename):
    epochs = range(1, train_err.size+1)
    val_epochs = [x*100 for x in range(val_err.size)]
    plt.plot(epochs, train_err, 'b', label='Training loss')
    plt.plot(val_epochs, val_err, 'g', label='Validation loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)

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
    model_conf = config["model"]
    activation = model_conf["activation_units"]
    max_step   = model_conf["max_step"]
    check_step = model_conf["check_step"]
    epsilon    = model_conf["epsilon"]
    
    
    ### loding hyperparameters from config ###
    layers     = model_conf["hidden_units"]
    batch_size = model_conf["batch_size"]
    eta        = model_conf["eta"]
    lam        = model_conf["lambda"]
    alpha      = model_conf["alpha"]
    print(f"\neta: {eta}\nlambda: {lam}\nbatch_size={batch_size}\nalpha: {alpha}")

    ### training ###
    #it reads as nonlocal variables: dl, activation, checkstep, maxstep, epsilon    
    train_err, val_err, nn = train(layers, batch_size, eta, lam, alpha)


    ### printing results ###
    print(f"train_err: {np.array(train_err)}")
    test_error = accuracy (dl.get_partition_set('test'), nn)
    print(f"accuracy: {(test_error)*100}%") 

    ### plotting loss ###
    output_path= os.path.abspath(config["output_path"])
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    graph_path = os.path.abspath(config["graph_path"])
    if (not os.path.exists(graph_path)):
        os.makedirs(graph_path)
    graph_name = args.graph_name if args.graph_name is not None else "training_loss.png"
    create_graph(np.array(train_err), np.array(val_err), os.path.join(graph_path, graph_name))

    ### saving model ###
    #nn.save_model(os.path.join(output_path, "best_model.h5"))
    

