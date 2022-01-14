from matplotlib import pyplot as plt
import numpy as np
from MLP import MLP
from types import SimpleNamespace
from configuration import Configuration

# miscclass and accuracy compute nn.h, with a threshold
# mse and mee compute nn.forward, the direct output of the output layer
def empirical_error(NN, set, metric):
    error = 0
    if metric == "missclass":
        for pattern in set:
            error += np.max(abs(NN.h(pattern[0]) - pattern[1]))
        return error/len(set)
    if metric == "accuracy":
        for pattern in set:
            error += np.max(abs(NN.h(pattern[0]) - pattern[1]))
        return 1-error/len(set)
    if metric == "mse":
        for pattern in set:
            error += ((NN.forward(pattern[0]) - pattern[1])**2).sum() 
        return error/len(set)
    elif metric == "mee":
        for pattern in set:
            error += ((NN.forward(pattern[0]) - pattern[1])**2).sum()**1/2
        return error/len(set)
    else:
        raise NotImplementedError("unknown metric")

class History:
    def __init__(self, hyp, metrics, folds):
        self.hyperparameters = hyp
        self.plots = {
            (set, metric): [ [] for k in range(folds) ]
            for set in metrics.keys()
            for metric in metrics[set]
        }
        if hyp.eta_decay == -1:
            self.name = f"{hyp.layers}_{hyp.batch_size}_{hyp.eta}_nonvar_{hyp.lam}_{hyp.alpha}"
        else:
            self.name = f"{hyp.layers}_{hyp.batch_size}_{hyp.eta}_{hyp.eta_decay}_{hyp.lam}_{hyp.alpha}"


    
    def update_plots(self, nn, fold, **sets):
        """ This function accept kwargs, whose names MUST BE 
        THE SAME DATASETS OF THE CONFIG FILE.
        The name of each argument will be used as a key, and the
        value of the argument will be passed to empricial_error
        Example: history.update_plots(nn, train=TR, test=TS)
        """
        for set, metric in self.plots:
            error = empirical_error(nn, sets[set], metric)
            self.plots[set, metric][fold].append(error)
        # for set_name, set_value in sets.items():
        #     for metric in self.plots[set_name]:
        #         error = empirical_error(nn, set_value, metric)
        #         self.plots[set_name][metric][fold].append(error)

    def get_last_error (self, set, metric, fold):
        return self.plots[set, metric][fold][-1]
        

    # def plot(self, path):
    #     os.join(path, self.name)
    #     plt.plot(self.history)
    #     plt.plot(self.history[train])
    #     plt.plot(self.history[train però mee])
    #     plt.plot(self.history)
    # def plot(self, path):
    #     plt.title(f'Mean: {history["mean"]:.2f} +- Var: {history["variance"]**0.5:.2f}')
    #     plt.xlabel('Epochs')
    #     plt.yscale('log')
    #     plt.ylabel('Loss')
    #     for i, train in enumerate(history['training']):
    #         epochs = range(len(train))
    #         plt.plot(epochs, train, linestyle='-', label=f'Training_{i}_fold loss')
    #     for i, val in enumerate(history['validation']):
    #         epochs = [x*history['val_step'] for x in range(len(val))]
    #         plt.plot(epochs, val, linestyle='--', label=f'Validation_{i}_fold loss')

    #     # #val_path = os.path.join(graph_path, 'validation')
    #     train_path = os.path.join(self.graph_path, 'training')
    #     if (not os.path.exists(train_path)):
    #         os.makedirs(train_path)
    #     plt.legend()
    #     plt.savefig(os.path.join(train_path, filename))
    #     plt.clf()

    #     plt.title(f'Average Weights change')
    #     plt.xlabel('Epochs')
    #     plt.yscale('log')
    #     plt.ylabel('Values')
    #     for i, wc in enumerate(history['weight_changes']):
    #         epochs = [x*history['val_step'] for x in range(len(val))]
    #         plt.plot(epochs, wc, linestyle='-.', label=f'WC_{i}_fold loss')
    #     grad_path = os.path.join(self.graph_path, 'gradients')
    #     if (not os.path.exists(grad_path)):
    #         os.makedirs(grad_path)
    #     plt.legend()
    #     plt.savefig(os.path.join(grad_path, filename))
    #     plt.clf()

    #     # plt.title(f'Validation Loss - AVG +- VAR')
    #     # plt.xlabel('Epochs')
    #     # plt.yscale('log')
    #     # plt.ylabel('Loss')
        
    #     # if (not os.path.exists(val_path)):
    #     #     os.makedirs(val_path)
    #     # plt.legend()
    #     # plt.savefig(os.path.join(val_path, filename))
    #     # plt.clf()


    # def create_final_graph (self, history, filename):
    #     plt.title(f'Train and Test error - Final Test MEE: {history["testing"]:.2f}')
    #     plt.xlabel('Epochs')
    #     plt.yscale('log')
    #     plt.ylabel('Loss')
    #     train = history['training']
    #     epochs = range(len(train))
    #     plt.plot(epochs, train, linestyle='-', label=f'Training loss')
    #     val = history['validation']
    #     epochs = [x*history['val_step'] for x in range(len(val))]
    #     plt.plot(epochs, val, linestyle='--', label=f'Test loss')

    #     # #val_path = os.path.join(graph_path, 'validation')
    #     train_path = os.path.join(self.graph_path, 'training')
    #     if (not os.path.exists(train_path)):
    #         os.makedirs(train_path)
    #     plt.legend()
    #     plt.savefig(os.path.join(train_path, filename))
    #     plt.clf()

    #     plt.title(f'Average Weights change')
    #     plt.xlabel('Epochs')
    #     plt.yscale('log')
    #     plt.ylabel('Values')
    #     wc = history['weight_changes']
    #     epochs = [x*history['val_step'] for x in range(len(val))]
    #     plt.plot(epochs, wc, linestyle='-.', label=f'WC loss')
    #     grad_path = os.path.join(self.graph_path, 'gradients')
    #     if (not os.path.exists(grad_path)):
    #         os.makedirs(grad_path)
    #     plt.legend()
    #     plt.savefig(os.path.join(grad_path, filename))
    #     plt.clf()

