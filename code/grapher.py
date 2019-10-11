import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def load_rslt():
    length_num = defaultdict()
    correct_num = defaultdict()
    with open('gen_accuracy.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            words = l.split()
            cat = words[0]
            length = int(words[1].split('.')[1])
            if length_num.get(length, 0) == 0:
                length_num[length] = 0
            length_num[length] += 1
            if correct_num.get(length, 0) == 0:
                correct_num[length] = 0
            if cat == 'gen':
                correct_num[length] += 1
        f.close()

    with open('spam_accuracy.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            words = l.split()
            cat = words[0]
            length = int(words[1].split('.')[1])
            if length_num.get(length, 0) == 0:
                length_num[length] = 0
            length_num[length] += 1
            if correct_num.get(length, 0) == 0:
                correct_num[length] = 0
            if cat == 'spam':
                correct_num[length] += 1
        f.close()

    rslt = defaultdict()
    for k in length_num.keys():
        total = length_num[k]
        correct = correct_num[k]
        rslt[k] = correct/total
    #print(dict(rslt))
    return dict(rslt)


def graph(d: dict):
    x, y = list(), list()
    for k in d.keys():
        x.append(k)
        y.append(d[k])
    #x, y = np.array(x)[np.newaxis, :], np.array(y)[np.newaxis,:]
    print(x)
    print(y)
    plt.scatter(x, y)
    plt.xlabel('length')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    #graph(load_rslt())
    x = [1, 2, 4, 8]
    y = [149/270, 149/270, 152/270, 149/270]
    plt.scatter(x, y)
    plt.xlabel('train file length')
    plt.ylabel('Accuracy')
    plt.show()
