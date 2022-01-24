from matplotlib import pyplot as plt
import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt
import os

# miscclass and accuracy compute nn.h, with a threshold
# mse and mee compute nn.forward, the direct output of the output layer
def empirical_error(NN, set, metric):
    if len(set) == 0:
        return None
    error = 0
    if metric == "missclass":
        for pattern in set:
            error += np.max(abs(NN.classify(pattern[0]) - pattern[1]))
        return error/len(set)
    if metric == "accuracy":
        for pattern in set:
            error += np.max(abs(NN.classify(pattern[0]) - pattern[1]))
        return 1-error/len(set)
    if metric == "mse":
        for pattern in set:
            error += ((NN.predict(pattern[0]) - pattern[1])**2).sum()
        return error/len(set)
    elif metric == "mee":
        for pattern in set:
            error += ((NN.predict(pattern[0]) - pattern[1])**2).sum()**1/2
        return error/len(set)
    else:
        raise NotImplementedError("unknown metric")

class History:
    def __init__(self, metrics):
        self.plots = {
            (set, metric): []
            for set in metrics.keys()
            for metric in metrics[set]
        }

    def update_plots(self, nn, **sets):
        """ This function accept kwargs, whose names MUST BE
        THE SAME DATASETS OF THE CONFIG FILE.
        The name of each argument will be used as a key, and the
        value of the argument will be passed to empricial_error
        Example: history.update_plots(nn, train=TR, test=TS)
        """
        for set, metric in self.plots:
            error = empirical_error(nn, sets[set], metric)
            self.plots[set, metric].append(error)
        # for set_name, set_value in sets.items():
        #     for metric in self.plots[set_name]:
        #         error = empirical_error(nn, set_value, metric)
        #         self.plots[set_name][metric][fold].append(error)

    def get_last_error (self, set, metric):
        return self.plots[set, metric][-1]

    def plot_in_graph (self, plt, set, metric, fold=0):
        epochs = range(len(self.plots[set, metric]))
        if set == "test":
            linestyle='-.'
        elif set == "val":
            linestyle = '--'
        else:
            linestyle = '-'
        plt.plot(epochs, self.plots[set, metric], linestyle=linestyle, label=f'{set} {metric} {fold}_fold loss')

class Results ():
    def __init__(self, hyp, metrics):
        self.hyperparameters = hyp
        #metrics.values() is a list of list of metrics, with a double for we concat all these metrics togoether
        #and by using a set comprehension, all the duplicates are deleted
        self.distinct_metrics = sorted({metric for metric_list in metrics.values() for metric in metric_list})
        self.distinct_sets = sorted({set for set in metrics})
        self.histories = []
        self.results = {
            (set, metric): {"mean": 0, "variance": 0}
            for set in metrics.keys()
            for metric in metrics[set]
        }
        layers = ""
        for layer in hyp["layers"]:
            layers += f'{layer[0][0:3]}'+f'{layer[1]}'
        if hyp["eta_decay"] == -1:
            self.name = layers+f'_{hyp["batch_size"]}_{hyp["eta"]}_nonvar_{hyp["lam"]}_{hyp["alpha"]}'
        else:
            self.name = layers+f'_{hyp["eta"]}_{hyp["eta_decay"]}_{hyp["lam"]}_{hyp["alpha"]}'

    def add_history(self, history):
        self.histories.append(history)

    def calculate_mean (self, graph_path):
        for set, metric in self.results:
            for h in self.histories:
                final_error = h.get_last_error(set, metric)
                self.results[set, metric]["mean"] += final_error/len(self.histories)
                self.results[set, metric]["variance"] += (final_error**2)/len(self.histories)
            self.results[set, metric]['variance'] -= self.results[set, metric]['mean']**2
        
        train_path = os.path.join(graph_path, "training")
        if (not os.path.exists(train_path)):
            os.makedirs(train_path)
        print(self.results)
        filename = os.path.join(train_path, f"{self.name}_results.txt")
        with open(filename, "w") as f:
            f.write(str(self.results))
        return self.results

    def create_graph (self, graph_path):
        #plt.rcParams["figure.figsize"] = (10,7*len(self.distinct_metrics))
        plt.rcParams["figure.figsize"] = (10*len(self.distinct_metrics), 7)
        for n, metric in enumerate(self.distinct_metrics):
            # plt.subplot(len(self.distinct_metrics), 1, n+1)
            plt.subplot(1, len(self.distinct_metrics), n+1)
            if ("val", metric) in self.results:
                plt.title(f'{metric} - Mean: {self.results["val", metric]["mean"]:.3f} +-dev: {self.results["val", metric]["variance"]**0.5:.4f}')
            elif ("test", metric) in self.results:
                plt.title(f'{metric} - Mean: {self.results["test", metric]["mean"]:.3f} +- dev: {self.results["test", metric]["variance"]**0.5:.4f}')
            else:
                plt.title(f'{metric} - Mean: {self.results["train", metric]["mean"]:.3f} +- dev: {self.results["train", metric]["variance"]**0.5:.4f}')
            for i, h in enumerate(self.histories): #in questo ciclo per ogni storia (quindi per ogni k_fold), disegna un plot per ogni set, con metrica fissa.
                for set in self.distinct_sets:
                    plt.xlabel('Epochs')
                    plt.yscale('log')
                    plt.ylabel('Loss')
                    h.plot_in_graph(plt, set, metric, i)
                    plt.legend(prop={'size': 15})

        plt.suptitle(self.name)
        filename = f"{self.name}.png"
        train_path = os.path.join(graph_path, "training")
        if (not os.path.exists(train_path)):
            os.makedirs(train_path)
        plt.tight_layout()
        plt.savefig(os.path.join(train_path, filename))
        #plt.show()
        plt.clf()
