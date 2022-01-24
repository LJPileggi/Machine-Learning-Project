import csv

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

N = 10
M = 20

def Fourier_basis(x, k, N):
    a =  []
    i = 1
    while i <= N:
        a.append(np.sin(i*k*x/np.pi))
        i += 1
    a = np.array(a)
    return a

def Fourier_series(x, w, k):
    return (Fourier_basis(x, k, w.size)*w).sum()

if __name__ == '__main__':
    bb = np.random.uniform(-10., 10., N)
    print(bb)
    k_weight = [i**-2 for i in range(1, N+1)]
    coeff = np.array(bb*k_weight)
    
    error = np.random.normal(0., 0.5, M)

    xx = np.random.uniform(0, 15, M)
    xx.sort()
    y = [Fourier_series(x, coeff, 1.) for x in xx]
    
    with open('dataset2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(zip(xx, y+error))

    plt.errorbar(xx, y+error, marker='.', linestyle='', color='black')
    plt.show()
