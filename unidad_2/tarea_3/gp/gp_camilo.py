import numpy as np


class GaussianProcess:
	"Clase de Proceso Gaussiano general"
	def __init__(self,kernel=None,sigma_n:float=0.1):
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

		cov_grid = self.kernel(self.time,self.time) + 1e-5*np.eye(self.N) + self.sigma_n**2*np.eye(self.N)

		if self.x is None:  # no observations 
			self.mean = np.zeros_like(self.time)
			self.cov = cov_grid
		
		else:  # observations
			cov_obs = self.kernel(self.x,self.x) + 1e-5*np.eye(self.Nx) + self.sigma_n**2*np.eye(self.Nx)
			cov_star = self.kernel(self.time,self.x)
			self.mean = np.squeeze(cov_star@np.linalg.solve(cov_obs,self.y))
			self.cov =  cov_grid - (cov_star@np.linalg.solve(cov_obs,cov_star.T))

	def train(self,verbose=True):
		sigma_n, _ = self.kernel.update_params(self.x, self.y, self.sigma_n, self.Nx,verbose=verbose)
		self.sigma_n = sigma_n
		if verbose:
			self.kernel.show_hypers()
			print(f'\tsigma_n ={self.sigma_n}')
			print("--------------------------")

	def pred_dist(self,where):

		time = where
		N = len(where)

		cov_grid = self.kernel(time,time) + 1e-5*np.eye(N) + self.sigma_n**2*np.eye(N)

		cov_obs = self.kernel(self.x,self.x) + 1e-5*np.eye(self.Nx) + self.sigma_n**2*np.eye(self.Nx)
		cov_star = self.kernel(time,self.x)

		mean = np.squeeze(cov_star@np.linalg.solve(cov_obs,self.y))
		cov =  cov_grid - (cov_star@np.linalg.solve(cov_obs,cov_star.T))

		return mean, cov
	
	def nll(self,x=None,y=None):
		if x is None and y is None:
			y = self.y
			x = self.x
			Nx = self.Nx
		else:
			Nx = len(x)
		Gram = self.kernel(x,x)
		K = Gram + self.sigma_n**2*np.eye(Nx) + 1e-5*np.eye(Nx)
		(_, logdet) = np.linalg.slogdet(K)
		return 0.5*(y.T@np.linalg.solve(K,y) + logdet + self.Nx*np.log(2*np.pi))
