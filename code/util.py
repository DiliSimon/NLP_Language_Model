import math
import numpy as np


def combine_vecs(vec, first, second, third):
    rslt = np.array(vec[first]) - np.array(vec[second]) + np.array(vec[third])
    rslt = rslt.tolist()
    return rslt


def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    sum = 0

    for idx, num_x in enumerate(x):
        num_y = y[idx]
        sum+=num_x*num_y
    sqrt_sum_x = 0
    sqrt_sum_y = 0
    for num_x in x:
        sqrt_sum_x += num_x**2
    norm_x = math.sqrt(sqrt_sum_x)
    for num_y in y:
        sqrt_sum_y += num_y**2
    norm_y = math.sqrt(sqrt_sum_y)
    sim = sum/(norm_x*norm_y + 1)
    return sim
