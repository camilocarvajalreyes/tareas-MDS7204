import numpy as np
from scipy.optimize import minimize

def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)

def Periodic(x,y,sigma,p,gamma):
	"""Implementacion de un kernel periodico"""
	return sigma**2 * np.exp((-2*(np.sin(np.pi*np.abs(x-y)/p))**2)/(gamma**2))


class SpectralMixtureKernel:
    """Kernel Mixtura Espectral"""
    def __init__(self,sigma=10,gamma=0.5,mu=0):
        self.sigma = sigma
        self.gamma = gamma
        self.mu = mu
    
    def show_hypers(self):
        print('Hyperparámetros Kernel SpectralMixture:')
        print(f'\tsigma = {self.sigma}')
        print(f'\tgamma = {self.gamma}')
        print(f'\tmu = {self.mu}')
    
    def __call__(self, X, Y):
        return self.sigma**2 * np.exp(-self.gamma*outersum(X,-Y)**2)*np.cos(2*np.pi*self.mu*outersum(X,-Y))
    
    @staticmethod
    def Spec_Mix(x,y,gamma, mu, sigma):
        return sigma**2 * np.exp(-gamma*outersum(x,-x)**2)*np.cos(2*np.pi*mu*outersum(x,-y))
    
    @staticmethod
    def Spec_Mix_sine(x, y, gamma, mu, sigma):
        return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.sin(2*np.pi*mu*outersum(x,-y))
    
    def nlogp(self,x,y,Nx,hypers):
        sigma = np.exp(hypers[0])
        gamma = np.exp(hypers[1])
        mu = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Gram = self.Spec_Mix(x,x,gamma, mu, sigma)
        K = Gram + sigma_n**2*np.eye(Nx) + 1e-5*np.eye(Nx)
        (_, logdet) = np.linalg.slogdet(K)
        return 0.5*(y.T@np.linalg.solve(K,y) + logdet + Nx*np.log(2*np.pi))

    def dnlogp(self,x,y,Nx, hypers):
        sigma = np.exp(hypers[0])
        gamma = np.exp(hypers[1])
        mu = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Gram = self.Spec_Mix(x,x,gamma,mu,sigma)
        K = Gram + sigma_n**2*np.eye(Nx) + 1e-5*np.eye(Nx)

        h = np.linalg.solve(K,y).T

        dKdsigma = 2*Gram/sigma
        dKdgamma = -Gram*(outersum(x,-x)**2)
        dKdmu = -2*np.pi*self.Spec_Mix_sine(x,x, gamma, mu, sigma)*outersum(x,-x)
        dKdsigma_n = 2*sigma_n*np.eye(Nx)

        H = (np.outer(h,h) - np.linalg.inv(K))
        dlogp_dsigma = sigma * 0.5*np.trace(H@dKdsigma)
        dlogp_dgamma = gamma * 0.5*np.trace(H@dKdgamma)
        dlogp_dmu = mu * 0.5*np.trace(H@dKdmu)
        dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)
        return np.array([-dlogp_dsigma, -dlogp_dgamma, -dlogp_dmu, -dlogp_dsigma_n])
    
    
    def update_params(self, X, Y, sigma_n, Nx, verbose=True):
        if self.mu == 0:
            self.mu = 0.1
		
        hypers0 = np.array([np.log(self.sigma), np.log(self.gamma), np.log(self.mu), np.log(sigma_n)])

        jacobian = lambda hypers: self.dnlogp(X, Y, Nx, hypers)
        obj_fun = lambda hypers: self.nlogp(X, Y, Nx, hypers)

        res = minimize(obj_fun, hypers0, args=(), method='L-BFGS-B', jac=jacobian, options={'maxiter': 500, 'disp': verbose})

        self.sigma = np.exp(res.x[0])
        self.gamma = np.exp(res.x[1])
        self.mu = np.exp(res.x[2])
        new_sigma_n = np.exp(res.x[3])

        return new_sigma_n, np.array([self.sigma, self.mu, self.gamma])

class PeriodicKernel:
    """Kernel Mixtura Espectral"""
    def __init__(self,sigma=1.0,length_scale=1.0,periodicity=1.0):
        self.sigma = sigma
        self.length_scale = length_scale
        self.periodicity = periodicity
    
    def show_hypers(self):
        print('Hyperparámetros Kernel Periodico:')
        print(f'\tsigma = {self.sigma}')
        print(f'\tlargo de escala = {self.length_scale}')
        print(f'\tperiodicidad = {self.periodicity}')
    
    def __call__(self, X, Y):
        return self.sigma**2 * np.exp(-2 * (np.sin(np.pi / self.periodicity * outersum(X,-Y)) / self.length_scale) ** 2)
    
    @staticmethod
    def kernel_eval(x,y,sigma,length_scale,periodicity):
        return sigma**2 * np.exp (-2 * (np.sin(np.pi / periodicity * outersum(x,-y)) / length_scale) ** 2 )
    
    def nlogp(self,x,y,Nx,hypers):
        sigma = np.exp(hypers[0])
        length_scale = np.exp(hypers[1])
        periodicity = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Gram = self.kernel_eval(x,x,sigma,length_scale,periodicity)
        K = Gram + sigma_n**2*np.eye(Nx) + 1e-5*np.eye(Nx)
        (_, logdet) = np.linalg.slogdet(K)
        return 0.5*(y.T@np.linalg.solve(K,y) + logdet + Nx*np.log(2*np.pi))

    def dnlogp(self,x,y,Nx, hypers):
        sigma = np.exp(hypers[0])
        length_scale = np.exp(hypers[1])
        periodicity = np.exp(hypers[2])
        sigma_n = np.exp(hypers[3])

        Gram = self.kernel_eval(x,x,sigma,length_scale,periodicity)
        K = Gram + sigma_n**2*np.eye(Nx) + 1e-5*np.eye(Nx)

        h = np.linalg.solve(K,y).T

        arg = np.pi * outersum(x,-x) / periodicity
        sin, cos = np.sin(arg), np.cos(arg)

        dKdsigma = 2 * sigma * K
        dKdl = 4 * K * sin**2 / length_scale**3
        dKdperiod = 4 * K * sin / length_scale**2 * cos * arg / periodicity
        dKdsigma_n = 2*sigma_n*np.eye(Nx)

        H = (np.outer(h,h) - np.linalg.inv(K))
        dlogp_dsigma = length_scale * 0.5*np.trace(H@dKdsigma)
        dlogp_dl = length_scale * 0.5*np.trace(H@dKdl)
        dlogp_dperiod = periodicity * 0.5*np.trace(H@dKdperiod)
        dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)
        return np.array([-dlogp_dsigma, -dlogp_dl, -dlogp_dperiod, -dlogp_dsigma_n])
    
    
    def update_params(self, X, Y, sigma_n, Nx, verbose=True):
		
        hypers0 = np.array([np.log(self.sigma), np.log(self.length_scale), np.log(self.periodicity), np.log(sigma_n)])

        jacobian = lambda hypers: self.dnlogp(X, Y, Nx, hypers)
        obj_fun = lambda hypers: self.nlogp(X, Y, Nx, hypers)

        res = minimize(obj_fun, hypers0, args=(), method='Newton-CG', jac=jacobian, options={'maxiter': 500, 'disp': verbose})

        self.sigma = np.exp(res.x[0])
        self.length_scale = np.exp(res.x[1])
        self.periodicity = np.exp(res.x[2])
        new_sigma_n = np.exp(res.x[3])

        return new_sigma_n, np.array([self.sigma, self.length_scale, self.periodicity])
