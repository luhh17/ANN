import numpy as np


def save_list(a, name):
    a = np.array(a)
    np.save(name + '.npy', a)   # 保存为.npy格式


def load_list(name):
    a = np.load(name + '.npy')
    a = a.tolist()
    return a