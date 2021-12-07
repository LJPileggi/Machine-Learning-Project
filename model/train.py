from dataloader import DataLoader
from MLP import MLP

import numpy as np
import os
import argparse
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Train a model.")
parser.add_argument('--config_path',
                    help='path to config file')
parser.add_argument('--graph_name',
                    help='name of the loss graph you will generate')
parser.add_argument('--grid_search', dest='grid_search', action='store_true',
                    help='If you are going to do a grid_search')
parser.set_defaults(grid_search=False)

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

def create_graph (history, filename):
    epochs = range(1, history.size+1)
    plt.plot(epochs, history, 'b', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Check Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)

if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))

    train_set  = config["train_set"]
    test_set   = config["test_set"]
    output_path= os.path.abspath(config["output_path"])
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    graph_path = os.path.abspath(config["graph_path"])
    if (not os.path.exists(graph_path)):
        os.makedirs(graph_path)
    graph_name = args.graph_name if args.graph_name is not None else "training_loss.png"

    encoding   = config["preprocessing"]["1_hot_enc"]

    model_conf = config["model"]
    batch_size = model_conf["batch_size"]
    epsilon    = model_conf["epsilon"]
    eta        = model_conf["eta"]
    lam        = model_conf["lambda"]
    alpha      = model_conf["alpha"]
    max_step   = model_conf["max_step"]
    check_step = model_conf["check_step"]
    layers     = model_conf["hidden_units"]
    activation = model_conf["activation_units"]


    dl = DataLoader ()
    dl.load_data ("train", train_set, encoding)
    dl.load_data ("test", test_set, encoding)

    whole_TR= dl.get_training_set()
    #whatch out! if batch_size = -1, it becomes len(TR)
    batch_size = len(whole_TR) if batch_size == -1 else batch_size
    print(f"epsilon: {epsilon}\neta: {eta}\nlambda: {lam}\nbatch_size={batch_size}\nalpha: {alpha}")
    
    input_size = dl.get_input_size ()
    nn = MLP (input_size, layers, activation)
    err = np.inf
    
    train_err = []
    for i in range (max_step):
        for current_batch in dl.training_set_partition(batch_size):
            for pattern in current_batch:
                out = nn.forward(pattern[0])
                error = pattern[1] - out
                nn.backwards(error)
            #we are updating with eta/TS_size in order to compute LMS, not simply LS
            nn.update_weights(eta/len(whole_TR), lam, alpha)
        if(i % check_step == 0):
            err = MSE_over_network (whole_TR, nn)
            print (f"{i}: {err}")
            train_err.append(err)
            if (np.allclose(err, 0, atol=epsilon)):
                nn.save_model(os.path.join(output_path, "best_model.h5"))
                break

    print(f"train_err: {np.array(train_err)}")

    create_graph(np.array(train_err), os.path.join(graph_path, graph_name))
    test_error = accuracy (dl.get_test_set(), nn)
    print(f"accuracy: {(test_error)*100}%") 
    #this accuracy calculation is wrong, because the error is squared and averaged

