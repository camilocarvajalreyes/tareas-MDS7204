from sklearn.gaussian_process.kernels import Kernel
import numpy as np
from scipy.optimize import minimize

def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)

class BaseKernel:
    def __call__(self, X, Y=None):
        raise NotImplementedError
    
    def show_hypers(self):
        for a in dir(self):
            if not a.startswith('__') and not a.startswith('_'):
                try:
                    print(a + " = {}".format(self.kernel.__dict__[a]))
                except KeyError:
                    pass
    
    def load_from_sklearn(self,sklearn_obj:Kernel):
        """Load a kernel from scikitlearn and transform it into a BaseKernel object usable within gp_lite toolbox"""
        pass


class SpectralMixtureKernel(BaseKernel):

    def __init__(self,):
        self.sigma = 10
        self.gamma = 1/2
        self.mu = 0
    
    def __call__(self, X, Y=None, eval_gradient=False):
        K = self.sigma**2 * np.exp(-self.gamma*outersum(X,-Y)**2)*np.cos(2*np.pi*self.mu*outersum(X,-Y))
        if not eval_gradient:
            return K
        else:
            return K, self.dnlogp(X,Y)
    
    def Spec_Mix_sine(self,x, y):
        return self.sigma**2 * np.exp(-self.gamma*outersum(x,-y)**2)*np.sin(2*np.pi*self.mu*outersum(x,-y))
    
    def dnlogp(self, X, Y, sigma_n, Nx):
        Gram = self(X,X,self.gamma,self.mu,self.sigma)
        K = Gram + sigma_n**2*np.eye(Nx) + 1e-5*np.eye(Nx)
        h = np.linalg.solve(K,Y).T

        dKdsigma = 2*Gram/self.sigma
        dKdgamma = -Gram*(outersum(X,-X)**2)
        dKdmu = -2*np.pi*self.Spec_Mix_sine(X,X, self.gamma, self.mu, self.sigma)*outersum(X,-X)
        dKdsigma_n = 2*sigma_n*np.eye(Nx)

        H = (np.outer(h,h) - np.linalg.inv(K))
        dlogp_dsigma = self.sigma * 0.5*np.trace(H@dKdsigma)
        dlogp_dgamma = self.gamma * 0.5*np.trace(H@dKdgamma)
        dlogp_dmu = self.mu * 0.5*np.trace(H@dKdmu)
        dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)

        return np.dstack((-dlogp_dsigma, -dlogp_dgamma, -dlogp_dmu, -dlogp_dsigma_n))
    
    def update_params(self, obj_fun, X, Y, sigma_n, Nx):
        if self.mu == 0:
            self.mu = 0.1
		
        hypers0 = np.array([np.log(self.sigma), np.log(self.gamma), np.log(self.mu), np.log(self.sigma_n)])

        jacobian = self.dnlogp(X, Y, sigma_n, Nx)

        res = minimize(obj_fun, hypers0, args=(), method='L-BFGS-B', jac = jacobian, options={'maxiter': 500, 'disp': True})

        self.sigma = np.exp(res.x[0])
        self.gamma = np.exp(res.x[1])
        self.mu = np.exp(res.x[2])
        self.sigma_n = np.exp(res.x[3])
        self.theta = np.array([self.mu, self.gamma, self.sigma_n ])
