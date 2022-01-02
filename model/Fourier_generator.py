import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pylab as pl
import csv

N = 10
M = 400

def Fourier_basis(x, k, N):
    a = [1.]
    i = 1
    while i <= N:
        a.append(np.sin(i*k*x/np.pi))
        i += 1
    a = np.array(a)
    return a

def Fourier_series(x, w, k):
    return (Fourier_basis(x, k, w.size-1)*w).sum()

if __name__ == '__main__':
    bb = np.random.uniform(-10., 10., N+1)
    print(f'random coefficients: {bb}')
    k_weight = [i**-2 for i in range(1, N+1)]
    k_weight.insert(0, 1.)
    print(k_weight)
    coeff = np.array(bb*k_weight)
    print(f'weighted random coefficients: {coeff}')
    
    error = np.random.normal(0., 0.5, M)

    xx = np.random.uniform(0, 15, M)
    xx.sort()
    y = [Fourier_series(x, coeff, 1.) for x in xx]

    xx = np.array(xx)
    y = np.array(y)

    indices = np.arange(M)
    
    test_mask = np.zeros(M, dtype=bool)
    test_mask[np.random.choice(indices, 20)] = True
    train_index = indices[np.logical_not(test_mask)]
    test_index = indices[test_mask]

    with open(os.path.join('..', 'data', 'fourier.train'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i, (x, f_x) in enumerate(zip(xx[train_index], y[train_index])):
            writer.writerow([ i, x, f_x ])
    with open(os.path.join('..', 'data', 'fourier.test'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i, (x, f_x) in enumerate(zip(xx[test_index], y[test_index])):
            writer.writerow([ i, x, f_x ])

    plt.errorbar(xx, y+error, marker='.', linestyle='', color='black')
    plt.plot(xx, y, marker='', linestyle='-', color='red')
    plt.show()
