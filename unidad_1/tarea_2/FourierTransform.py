import numpy as np

def DiscreteFourierTransform(x,k):
    N = x.shape[0]
    Xk = np.sum(x * np.exp(-2j * np.pi * k * np.arange(N)/N))
    return Xk


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


def FFT(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    """
    N = len(x)
    
    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X
