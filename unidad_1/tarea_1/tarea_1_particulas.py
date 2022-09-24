from filtros.filtro_particulas import ParticleFilter1DStochasticVolatility
from plot import plot_series
import numpy as np

# parámetros de este código
np.random.seed(0)
save_plots = False  # si guardar o no las figuras
img_dir = "unidad_1/tarea_1/informe/img"  # directorio en caso de guardar las figuras
display_plots = True  # si se muestran o no las figuras (pop-up), no funciona al mismo tiempo que save_plots

# Simulación de las series

SIGMA, ALPHA = 1, 0.8
BETA = 0.5

X1 = np.random.normal(0,(SIGMA**2)/(1-(ALPHA**2)))

n_samples = 500

V = np.random.normal(0,1,n_samples)
W = np.random.normal(0,1,n_samples)


X = np.zeros(n_samples)
Y = np.zeros(n_samples)
X[0] = X1
for i in range(n_samples):
    try:
        X[i+1] = ALPHA*X[i] + SIGMA * V[i]
    except IndexError:
        pass
    Y[i] = BETA*np.exp(X[i]/2)*W[i]

if save_plots:
    plot_series([X,Y],['X','Y'],save='series_x_y.png',folder=img_dir,display=display_plots)
    # plot_series(X,'X',save='serie_x.png',folder=img_dir,display=display_plots)
    # plot_series(Y,'Y','tab:orange',save='serie_y.png',folder=img_dir,display=display_plots)
else:
    plot_series([X,Y],['X','Y'],display=display_plots)
    # plot_series(X,'X',display=display_plots)
    # plot_series(Y,'Y','tab:orange',display=display_plots)

# Filtro de partículas
