import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import os

class DataStorage:
    
    def __init__(self, data_conf):
        now = datetime.now()
        date = str(datetime.date(now))
        time = str(datetime.time(now))
        time = time[:2] + time[3:5]
        
        output_path = os.path.abspath(data_conf["output_path"])
        output_path = os.path.join(output_path, date, time)
        self.output_path = output_path
        print(output_path)
        if (not os.path.exists(output_path)):
            os.makedirs(output_path)
        
        graph_path = os.path.abspath(data_conf["graph_path"])
        graph_path = os.path.join(graph_path, date, time)
        self.graph_path = graph_path
        print(graph_path)
        if (not os.path.exists(graph_path)):
            os.makedirs(graph_path)

        
    def save_model(self, nn, filename):
        path = os.path.join(self.output_path, filename)
        nn.save_model(path)


    def create_graph (self, history, filename):
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
        train_path = os.path.join(self.graph_path, 'training')
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
        grad_path = os.path.join(self.graph_path, 'gradients')
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


    def create_final_graph (self, history, filename):
        plt.title(f'Train and Test error - Final Test MEE: {history["testing"]:.2f}')
        plt.xlabel('Epochs')
        plt.yscale('log')
        plt.ylabel('Loss')
        train = history['training']
        epochs = range(len(train))
        plt.plot(epochs, train, linestyle='-', label=f'Training loss')
        val = history['validation']
        epochs = [x*history['val_step'] for x in range(len(val))]
        plt.plot(epochs, val, linestyle='--', label=f'Test loss')

        # #val_path = os.path.join(graph_path, 'validation')
        train_path = os.path.join(self.graph_path, 'training')
        if (not os.path.exists(train_path)):
            os.makedirs(train_path)
        plt.legend()
        plt.savefig(os.path.join(train_path, filename))
        plt.clf()

        plt.title(f'Average Weights change')
        plt.xlabel('Epochs')
        plt.yscale('log')
        plt.ylabel('Values')
        wc = history['weight_changes']
        epochs = [x*history['val_step'] for x in range(len(val))]
        plt.plot(epochs, wc, linestyle='-.', label=f'WC loss')
        grad_path = os.path.join(self.graph_path, 'gradients')
        if (not os.path.exists(grad_path)):
            os.makedirs(grad_path)
        plt.legend()
        plt.savefig(os.path.join(grad_path, filename))
        plt.clf()
