import numpy as np
import matplotlib.pyplot as plt

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

def create_graph (history, graph_path, filename):
    plt.title(f'Train and Validation error - Mean: {history["mean"]:.2f} +- Var: {history["variance"]**0.5:.2f}')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.ylabel('Loss')
    for i, train in enumerate(history['training']):
        epochs = range(len(train))
        plt.plot(epochs, train, linestyle='-', label=f'Training_{i}_fold loss')
    for i, val in enumerate(history['validation']):
        epochs = [x*history['val_step'] for x in range(len(val))]
        plt.plot(epochs, val, linestyle='--', label=f'Validation_{i}_fold loss')

    # #val_path = os.path.join(graph_path, 'validation')
    train_path = os.path.join(graph_path, 'training')
    if (not os.path.exists(train_path)):
        os.makedirs(train_path)
    plt.legend()
    plt.savefig(os.path.join(train_path, filename))
    plt.clf()

    plt.title(f'Average Weights change')
    plt.xlabel('Epochs')
    plt.yscale('log')
    plt.ylabel('Values')
    for i, wc in enumerate(history['weight_changes']):
        epochs = [x*history['val_step'] for x in range(len(val))]
        plt.plot(epochs, wc, linestyle='-.', label=f'WC_{i}_fold loss')
    grad_path = os.path.join(graph_path, 'gradients')
    if (not os.path.exists(grad_path)):
        os.makedirs(grad_path)
    plt.legend()
    plt.savefig(os.path.join(grad_path, filename))
    plt.clf()

    # plt.title(f'Validation Loss - AVG +- VAR')
    # plt.xlabel('Epochs')
    # plt.yscale('log')
    # plt.ylabel('Loss')
    
    # if (not os.path.exists(val_path)):
    #     os.makedirs(val_path)
    # plt.legend()
    # plt.savefig(os.path.join(val_path, filename))
    # plt.clf()
