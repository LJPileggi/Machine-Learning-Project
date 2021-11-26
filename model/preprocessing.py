import numpy as np

f = open("monks-1.test", "r")

data = []
for line in f.readline:
    l = list(map(int, line.split()))
    data.append( (np.array(ele[1:]), np.array(ele[0]) ) for ele in l )

print(data)
