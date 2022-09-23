import numpy as np

class Distribution:
    def __init__(self):
        pass

    def sample(self,N):
        samples = None
        return samples


class SequentialImportanceSampling:
    def __init__(self,N):
        self.particle_sequence = []
        self.weights_sequence = []
        self.N = N

    def sample_particles(self,dist):
        new_particles = dist.sample(self.N)
        self.particle_sequence.append(new_particles)

    def update_weights(self,y_obs):
        last_weights = self.weights_sequence[-1]
        assert y_obs.shape[0] == last_weights.shape[0]
        sum_w, new_weights = 0, []
        for i, w in enumerate(last_weights):
            particula = self.particle_sequence[-1][i]
            new_w = w * np.exp(-0.5*(np.exp(-1*particula)*y_obs[i]**2 + particula))
            sum_w += new_w
            new_weights.append(new_w)
        new_weights_norm = np.array([w/sum_w for w in new_weights])
        self.weights_sequence.append(new_weights_norm)


class ParticleFilter1D:
    def __init__(self,mu_inicial,N):
        """Inicialización de la clase
        
        Argumentos
        ----------
        
            mu_inicial: Distribution instance
                distribucion para el estado inicial
                
            N: int
                numero de sampleos por paso
                
        """
        self._observed = []
        self._medias = []
        self.SIS = SequentialImportanceSampling(N)

        self.mu_inicial = mu_inicial

        self.SIS.sample_particles(self.mu_inicial)
        self.SIS.weights_sequence.append(np.ones(N)/N)
    
    def observe(self,y_obs):
        self._observed.append(y_obs)


    def iterar(self):
        """Método que lleva a cabo un paso del algoritmo"""
        new_dist = Distribution()  # inferir distribución

        particulas = self.SIS.sample_particles(new_dist)
        pesos = self.SIS.update_weights(self._observed[-1])
        self._medias.append(np.average(particulas,weights=pesos))
    
    # métodos getter
    @property
    def medias(self):
        return np.array(self._medias)
