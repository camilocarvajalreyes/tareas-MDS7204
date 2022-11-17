import numpy as np
from kernels import SpectralMixtureKernel

def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)

def Spec_Mix(x,y, gamma, mu, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.cos(2*np.pi*mu*outersum(x,-y))

def Periodic(x,y,sigma,p,gamma):
	"""Implementacion de un kernel periodico"""
	return sigma**2 * np.exp((-2*(np.sin(np.pi*np.abs(x-y)/p))**2)/(gamma**2))

def Spec_Mix_sine(x,y, gamma, mu, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.sin(2*np.pi*mu*outersum(x,-y))

class GaussianProcessBase:
	"Clase de Proceso Gaussiano general"
	def __init__(self,kernel:SpectralMixtureKernel=None,sigma_n:float=0.1):
		"""Inicialización de la clase con un Kernel arbitrario a definir"""
		self.kernel = kernel
		self.sigma_n = sigma_n

	def sample(self, how_many=1):
		samples =  np.random.multivariate_normal(self.mean, self.cov, size=how_many)
		self.samples = samples.T
		return self.samples

	def load(self, x, y):
		self.Nx = len(x)
		self.x = x
		self.y = y
	
	def compute_posterior(self, where = None):
		
		if where is not None:
			self.time = where
			self.N = len(where)

		cov_grid = self.kernel(self.time,self.time, self.gamma, self.mu, self.sigma) + 1e-5*np.eye(self.N) + self.sigma_n**2*np.eye(self.N)

		if self.x is None: #no observations 
			self.mean = np.zeros_like(self.time)
			self.cov = cov_grid
		
		else:  # observations
			cov_obs = self.kernel(self.x,self.x,self.gamma,self.mu,self.sigma) + 1e-5*np.eye(self.Nx) + self.sigma_n**2*np.eye(self.Nx)
			cov_star = self.kernel(self.time,self.x, self.gamma, self.mu, self.sigma)
			self.mean = np.squeeze(cov_star@np.linalg.solve(cov_obs,self.y))
			self.cov =  cov_grid - (cov_star@np.linalg.solve(cov_obs,cov_star.T))

	def nlogp(self):
		Y = self.y
		Gram = self.kernel(self.x,self.x)
		K = Gram + self.sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
		(_, logdet) = np.linalg.slogdet(K)
		return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Nx*np.log(2*np.pi))

	def nll(self):
		Y = self.y
		Gram = self.kernel(self.x,self.x,self.gamma,self.mu,self.sigma)
		K = Gram + self.sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
		(_, logdet) = np.linalg.slogdet(K)
		return 0.5*(Y.T@np.linalg.solve(K,Y) + logdet + self.Nx*np.log(2*np.pi))

	"""def dnlogp(self, hypers):
		sigma = np.exp(hypers[0])
		gamma = np.exp(hypers[1])
		mu = np.exp(hypers[2])
		sigma_n = np.exp(hypers[3])

		Y = self.y
		Gram = Spec_Mix(self.x,self.x,gamma,mu,sigma)
		K = Gram + sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
		h = np.linalg.solve(K,Y).T

		dKdsigma = 2*Gram/sigma
		dKdgamma = -Gram*(outersum(self.x,-self.x)**2)
		dKdmu = -2*np.pi*Spec_Mix_sine(self.x,self.x, gamma, mu, sigma)*outersum(self.x,-self.x)
		dKdsigma_n = 2*sigma_n*np.eye(self.Nx)

		H = (np.outer(h,h) - np.linalg.inv(K))
		dlogp_dsigma = sigma * 0.5*np.trace(H@dKdsigma)
		dlogp_dgamma = gamma * 0.5*np.trace(H@dKdgamma)
		dlogp_dmu = mu * 0.5*np.trace(H@dKdmu)
		dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)
		return np.array([-dlogp_dsigma, -dlogp_dgamma, -dlogp_dmu, -dlogp_dsigma_n])"""

	def train(self, flag = 'quiet'):
		self.kernel.update_params(self.nlogp)
		if flag != 'quiet':
			print('Hyperparameters are:')
			self.kernel.show_hypers()

	def pred_dist(self,where):

		time = where
		N = len(where)

		cov_grid = Spec_Mix(time,time, self.gamma, self.mu, self.sigma) + 1e-5*np.eye(N) + self.sigma_n**2*np.eye(N)

		cov_obs = Spec_Mix(self.x,self.x,self.gamma,self.mu,self.sigma) + 1e-5*np.eye(self.Nx) + self.sigma_n**2*np.eye(self.Nx)
		cov_star = Spec_Mix(time,self.x, self.gamma, self.mu, self.sigma)

		mean = np.squeeze(cov_star@np.linalg.solve(cov_obs,self.y))
		cov =  cov_grid - (cov_star@np.linalg.solve(cov_obs,cov_star.T))

		return mean, cov
