import numpy as np

def DiscreteFourierTransform(x):
    N = x.shape[0]
    X = np.zeros_like(x)
    for k in range(N):
        X[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(N)/N))
    return X


def FastFourierTransform(x):
    N = x.shape[0]
    X = np.zeros_like(x)
    if N==1:
        X[0] = x
        return X
    else:
        X_par = FastFourierTransform(x[slice(0,len(x),2)])
        X_impar = FastFourierTransform(x[slice(1,len(x),2)])
        mitad = int((N/2))
        X = np.concatenate(
            (X_par+np.exp(-2j*np.pi*np.arange(mitad)/ N)*X_impar,
            X_par+np.exp(-2j*np.pi*np.arange(mitad,N)/ N)*X_impar))
        return X
