from sklearn.gaussian_process.kernels import Kernel
import numpy as np

def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)

class SpectralMixtureKernel(Kernel):

    def __init__(self,):
        self.sigma = 10
        self.gamma = 1/2
        self.mu = 0
        self.sigma_n = 0.1
    
    def __call__(self, X, Y=None, eval_gradient=False):
        K = self.sigma**2 * np.exp(-self.gamma*outersum(X,-Y)**2)*np.cos(2*np.pi*self.mu*outersum(X,-Y))
        if not eval_gradient:
            return K
        else:
            return K, self.dnlogp(X,Y)
    
    def Spec_Mix_sine(self,x, y):
        return self.sigma**2 * np.exp(-self.gamma*outersum(x,-y)**2)*np.sin(2*np.pi*self.mu*outersum(x,-y))
    
    def dnlogp(self, X, Y):
        Gram = self(X,X,self.gamma,self.mu,self.sigma)
        K = Gram + self.sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
        h = np.linalg.solve(K,Y).T

        dKdsigma = 2*Gram/self.sigma
        dKdgamma = -Gram*(outersum(X,-X)**2)
        dKdmu = -2*np.pi*self.Spec_Mix_sine(X,X, self.gamma, self.mu, self.sigma)*outersum(X,-X)
        dKdsigma_n = 2*self.sigma_n*np.eye(self.Nx)

        H = (np.outer(h,h) - np.linalg.inv(K))
        dlogp_dsigma = self.sigma * 0.5*np.trace(H@dKdsigma)
        dlogp_dgamma = self.gamma * 0.5*np.trace(H@dKdgamma)
        dlogp_dmu = self.mu * 0.5*np.trace(H@dKdmu)
        dlogp_dsigma_n = self.sigma_n * 0.5*np.trace(H@dKdsigma_n)

        return np.dstack((-dlogp_dsigma, -dlogp_dgamma, -dlogp_dmu, -dlogp_dsigma_n))
