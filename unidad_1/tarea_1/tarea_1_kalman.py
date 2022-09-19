import numpy as np
from plot import plot_series
from filtros.filtro_kalman import KalmanFilter1D

# parámetros de este código
np.random.seed(0)

# Simulación de las series

print("Sampleando los parámetros")
A = np.random.uniform(0,1)
B = np.random.uniform(0.8,1.2)
C = np.random.uniform(0,1)
D = np.random.uniform(0.3,0.7)
X0 = np.random.normal(0,0.8)

print("Valor de A = {}\nValor de B = {}\nValor de C = {}\nValor de D = {}\nPosición inicial = {}".format(A,B,C,D,X0))

n_samples = 200

V = np.random.normal(0,1,n_samples)
W = np.random.normal(0,1,n_samples)

X = np.zeros(n_samples)
Y = np.zeros(n_samples)
X[0] = X0
for i in range(n_samples):
    try:
        X[i+1] = A*X[i] + B * V[i]
    except IndexError:
        pass
    Y[i] = C*X[i] + D*W[i]

plot_series(X,'X')
plot_series(Y,'Y','tab:orange')

# Filtro de Kalman

# definiendo el filtro
filter = KalmanFilter1D(Y[0],B*D,A,B,C,D)

# aplicación manual de las primeras iteraciones
filter.predict()
filter.x_predicted()
filter.observe(Y[1])
# print(filter._x_observed)
filter.update()
# print(filter.x_updated())

# iterando sobre todas las observaciones

for i in range(2,n_samples):
    # predicción
    filter.predict()
    # observación
    filter.observe(Y[i])
    # actualizacion
    filter.update()

# Visualización
X_pred = filter.x_predicted()
# plot_series(X_pred,'X predicted')

X_updt = filter.x_updated()
# plot_series(X_updt,'X updated')

plot_series([X,X_updt],['X','X updated'])

plot_series([X,X_pred,X_updt],['X','X predicted','X updated'])
