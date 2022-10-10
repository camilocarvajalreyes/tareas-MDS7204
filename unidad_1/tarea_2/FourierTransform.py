import numpy as np

def DiscreteFourierTransform(x,k):
    N = x.shape[0]
    Xk = np.sum(x * np.exp(-2j * np.pi * k * np.arange(N)/N))
    return Xk


def FastFourierTransform(x,N,s):
    X = np.zeros_like(x)
    if N==1:
        return [x[0]]
    else:
        half = int((N/2))
        X[:half-1] = FastFourierTransform(x,half,2*s)
        X[half-1:] = FastFourierTransform(x+s,half,2*s)
        for k in range(half):
            p = X[k]
            q = np.exp(-2*k*np.pi*1j/N)*X[k+half]
            X[k] = p + q
            X[k+half] = p - q
