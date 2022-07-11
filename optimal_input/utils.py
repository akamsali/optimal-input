import numpy as np

def corr(A, B):
    B_z = np.pad(B, ((A.shape[0] - 1,0),(0,0)), 'constant')

    corr = np.zeros(B_z.shape)
    for i in range(B_z.shape[0] - A.shape[0]):
        corr[i] = np.sum(A * B_z[i : i + A.shape[0], :])

    return corr