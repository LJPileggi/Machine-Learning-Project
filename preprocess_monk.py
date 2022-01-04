import csv
import os


def one_hot_enc(inputs):
    domains = [
        ['1', '2', '3'],
        ['1', '2', '3'],
        ['1', '2'],
        ['1', '2', '3'],
        ['1', '2', '3', '4'],
        ['1', '2' ]
    ]
    encoded = []
    for input, dom in zip(inputs, domains):
        if not (input in dom):
            raise Exception("Incorrect encoding!")
        vector = [(1 if input == val else 0) for val in dom]
        encoded.extend(vector)
    return encoded



if __name__ == "__main__":
    filenames = ["data/monks-1.train", "data/monks-2.train", "data/monks-3.train", ]
    for filename in filenames:
        f_in  = open (filename, "r", newline='')
        f_out = open (filename+".csv", "w", newline='')
        writer = csv.writer(f_out)
        for line in f_in.readlines():
                data = line.split()
                id = [data[-1]]
                inputs = data[1:-1]
                inputs = one_hot_enc(inputs)
                outputs = [data[0]]
                writer.writerow(id+inputs+outputs)
        print("done")
                