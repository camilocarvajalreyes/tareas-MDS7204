# FFT vs DFT

> Identifique la complejidad computacional tanto de DFT como de FFT. Explique las diferencias y los supuestos que hace FFT que le permiten realizar el calculo de la transformada de Fourier de forma mas eficiente.

## Discrete Fourier Transform

**Complejidad**: $O(N^2)$

$$ X_k = \sum^{N-1}_{n=1}x_n e^{-j \frac{2\pi}{N} kn} = \sum^{N-1}_{n=1}x_n[\cos(\frac{2\pi}{N} kn)-j\sin(\frac{2\pi}{N} kn)] $$

## Fast Fourier Transform

**Complejidad**: $O(N\log(N))$

**Pseudocódigo**:

Dados $x_0,X_s,X_{2s}\dots,x_{(N-1)s}$, calculamos su DFT ditfft2(x, N, s) = $X_0,\dots,X_{N-1}$ como 

    if N = 1 then
        X0 ← x0                                      trivial size-1 DFT
        base case
    else
        X0,...,N/2−1 ← ditfft2(x, N/2, 2s)             DFT of (x0, x2s, x4s, ..., x(N-2)s)
        XN/2,...,N−1 ← ditfft2(x+s, N/2, 2s)           DFT of (xs, xs+2s, xs+4s, ..., x(N-1)s)
        for k = 0 to N/2−1 do                        combine DFTs of two halves into full DFT:
            p ← Xk
            q ← exp(−2πi/N k) Xk+N/2
            Xk ← p + q 
            Xk+N/2 ← p − q
        end for
    end if
