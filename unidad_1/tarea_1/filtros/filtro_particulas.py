import numpy as np
from scipy.stats import norm, rv_discrete


class ParticleFilter1D:
    def __init__(self,N,resample=True):
        """Inicialización de la clase de un Filtro de una dimensión genérico
            Incluye los métodos observar, paso y resampling
            Los métodos específicos del modelo deben ser implementados: sample_particles y update_weights
        
        Argumentos
        ----------
                
            N: int
                numero de particulas por paso

            resample: bool
                si se implementa SIR en vez de SIS
                
        """
        self.N = N
        self.resample = resample
        self.particle_sequence = []
        self.weights_sequence = []
        self.weights_sequence.append(np.ones(self.N)/self.N)
        self._observed = []
        self._medias = []
        self._params = None

    def observe(self,y_obs):
        "Observa una realización de la variable observable Y"
        self._observed.append(y_obs)

    def sample_particles(self):
        raise NotImplementedError

    def update_weights(self):
        raise NotImplementedError

    def resampling(self):
        diracs = self.particle_sequence.pop()
        weights = self.weights_sequence.pop()
        ind_particles = rv_discrete(values=(np.arange(self.N,weights))).rvs(size=self.N)
        new_particles = diracs[ind_particles]
        self.particle_sequence.append(new_particles)

    def step(self):
        "Método que lleva a cabo un paso del algoritmo"
        self.sample_particles()
        self.update_weights(self._observed[-1])
        if self.resample:
            self.resampling()
        self._medias.append(np.average(self.particle_sequence[-1],weights=self.weights_sequence[-1]))
    
    # métodos getter
    @property
    def medias(self):
        return np.array(self._medias)


class ParticleFilter1DStochasticVolatility(ParticleFilter1D):
    """Clase que hereda de un filtro de particulas genérico (ParticleFilter1D)
    
    Específico para un modelo de volatilidad estocástica"""
    def set_initial_particles(self,mu,sigma):
        """
        Samplea segun una distribución normal N(mu,sigma)
        
        Arguments
        ----------
        
            mu: float
                media de la normal a considerar

            sigma: float
                desviacion estandar (sigma^2 en estricto rigor)

        """
        self.particle_sequence.append(norm.rvs(mu,sigma,size=self.N))
    
    def sv_params(self,params_dict):
        """Recibe los parametros de un modelo de volatilidad estocástica de la forma:
            Xn = alpha*Xn-1 + sigma^2 Vn + nu
            Yn = beta*exp(Xn/gamma)*Wn
        con Vn y Wn ruidos Gaussianos (normal estandar).

        Arguments
        ----------
        
            params_dict: dict
                diccionario con llaves 'sigma', 'alpha', 'beta', 'gamma', 'nu
        
        """
        s_key = 'sigma' in params_dict.keys()
        a_key = 'alpha' in params_dict.keys()
        b_key = 'beta' in params_dict.keys()
        g_key = 'gamma' in params_dict.keys()
        v_key = 'nu' in params_dict.keys()
        if not s_key and a_key and b_key and g_key and v_key:
            raise KeyError("'sigma', 'alpha', 'beta', 'gamma' and 'nu expected as keys of the parameters dictionary")
        self._params = params_dict

    @staticmethod
    def sample_gaussian(mu,sigma):
        new_particle = norm.rvs(mu,sigma,size=1)
        return new_particle

    def sample_particles(self):
        old_particles =  self.particle_sequence[-1]
        new_particles = []
        for part in old_particles:
            mu = part * self._params['alpha'] + self._params['nu']
            new_particles.append(norm.rvs(mu,self.sigma**2))

        self.particle_sequence.append(np.array(new_particles))

    def update_weights(self,y_obs):
        last_weights = self.weights_sequence[-1]
        assert y_obs.shape[0] == last_weights.shape[0]
        sum_w, new_weights = 0, []
        for i, w in enumerate(last_weights):
            particula = self.particle_sequence[-1][i]
            var = self._params['beta'] * np.exp(particula/self._params['gamma'])
            new_w = w * norm.pdf(self._observed[-1],scale=var)
            sum_w += new_w
            new_weights.append(new_w)
        new_weights_norm = np.array([w/sum_w for w in new_weights])
        self.weights_sequence.append(new_weights_norm)
