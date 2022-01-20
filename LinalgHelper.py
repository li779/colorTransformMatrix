import numpy as np

def normalize(data):
    shift = np.average(data,axis=0)
    data_centered = data-shift
    scale = np.max(np.linalg.norm(data_centered, axis=1))
    T = [[1/scale, 0, 0, -shift[0]/scale],[0, 1/scale, 0, -shift[1]/scale],[0, 0, 1/scale, -shift[2]/scale],[0, 0, 0, 1]]
    T = np.array(T)
    return T