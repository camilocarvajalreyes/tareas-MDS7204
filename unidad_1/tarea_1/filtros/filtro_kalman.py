import numpy as np

class KalmanFilter1D:
    def __init__(self,x0,P0,F_model,Q_model,H_model,sigma):
        """Inicializa la clase con listas vacías de observación, predicción y corrección
        
        Argumentos
        ----------
        
            x0: float
                estado inicial
                
            P0: float
                varianza inicial
                
            F_model: float
                factor que multiplica x_{n-1} en x_n

            Q_model: float
                varianza del ruido de la ecuación de estado
                
            H_model: float
                factor que multuplica x_n en y_n

            sigma: float
                varianza del ruido de medición
        
        """
        self._x_observed = [x0]
        # self.x_true = []
        self._x_predicted = []
        self._x_updated = []

        self._variances_predicted = []
        self._variances_updated = [P0]

        self.F = F_model
        self.Q = Q_model
        self.H = H_model
        self.sigma = sigma

        self._K = []

    def predict(self):
        x_pred = self.F * self._x_observed[-1]
        self._x_predicted.append(x_pred)
        P_pred = (self.F**2) * self._variances_updated[-1] + self.Q**2  # o self.Q ???
        self._variances_predicted.append(P_pred)

    def observe(self,x_obs):
        self._x_observed.append(x_obs)

    def kalman_gain(self):
        P = self._variances_predicted[-1]
        self._K.append(P / (P + (self.sigma**2)))

    def update(self):
        y = self._x_observed[-1] - self.H * self._x_predicted[-1]
        self.kalman_gain()
        x_updt = self._x_predicted[-1] + self._K[-1] * y
        self._x_updated.append(x_updt)
        P_updt = (1 - self._K[-1]) * self._variances_predicted[-1]
        self._variances_updated.append(P_updt)

    # getters
    def x_predicted(self):
        return np.array(self._x_predicted)

    def x_updated(self):
        return np.array(self._x_updated)
