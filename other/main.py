import csv

import numpy as np

import LMS_grad_desc

N = 20
N_max = 100

def Powers_basis(x, n):
    a = []
    i = 0
    while i <= n:
        a.append(x**i)
    a = np.array(a)
    return a
    
def Powers_series(x, n, w):
    return (Powers_basis(x, n)*w).sum()


if __name__ == '__main__':
    """
    with open('dataset2.csv', newline='') as file:
        reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
        xx, yy = [], []
        for row in reader:
            xx.append(row[0])
            yy.append(row[1])
    """
    xx = [0.006551043651579769, 0.2983811654890506, 1.078571227520098, 2.741807528518432, 3.9380225468747883, 6.445576997540805, 6.836049328941829, 7.731965165580172, 8.533799334739362, 9.932066021189069, 9.949073153620434, 10.045747965474682, 10.306863163896493, 10.560005946651149, 13.427344175714483, 13.592057689585424, 13.620208869489712, 14.001222169130113, 14.082233480423648, 14.711523358588956]
    yy = [-0.6468087461152854, 3.6134259516491323, 3.5531810407694335, 3.708649037903622, 2.897867828869743, 0.25428410486808706, -0.06242953173528079, -0.15312835006475933, 1.7506262333120768, -0.38728204174645875, 0.30825554301491204, -1.0438957712605534, -0.6121624828792595, -0.7369672069917776, 0.02063825172205988, 0.07992172012296672, 0.8224124166730931, -0.6919453768756121, -0.6067413552807464, -2.7355254980985158]

    w = np.random.uniform(-0.5, 0.5, N+1)
    w_old = w
    for j in range(N_max):
        for x, y in zip(xx, yy):
            w = LMS_grad_desc(Powers_basis(x, N), w, y, N_max=1)
            print(w)
            if abs((w - w_old).sum()) < 0.001:
                break
            w_old = w
    
    XX = np.linspace(0, 15, 2000)
    Y = [Powers_series(x, N, w) for x in XX]
    plt.errorbar(xx, y+error, marker='.', linestyle='', color='black')
    plt.plot(XX, Y, marker='.', linestyle='', color='red')
    plt.show()
