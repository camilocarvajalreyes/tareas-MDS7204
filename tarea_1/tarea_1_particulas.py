from filtros.filtro_particulas import ParticleFilter1DStochasticVolatility
from plot import plot_series
import numpy as np

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m*60
    if m < 60:
        return '%dm %ds' % (m, s)
    else:
        h = math.floor(m/60)
        m -= h*60
        return '%dh %dm %ds' % (h, m, s)

# parámetros de este código
np.random.seed(0)
save_plots = False  # si guardar o no las figuras
img_dir = "unidad_1/tarea_1/informe/img"  # directorio en caso de guardar las figuras
display_plots = True  # si se muestran o no las figuras (pop-up), no funciona al mismo tiempo que save_plots
SIS = True
SIR = True
N = 1000

SIGMA, ALPHA = 1, 0.8
BETA = 0.5

""" Parametros Stochastic Volatily
    Xn = alpha*Xn-1 + sigma^2 Vn + nu
    Yn = beta*exp(Xn/gamma)*Wn
con Vn y Wn ruidos Gaussianos (normal estandar)."""
SV_params = {
    'sigma' : SIGMA,
    'alpha' : ALPHA,
    'beta' : BETA,
    'gamma' : 2,
    'nu' : 0
}

# Simulación de las series
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
    plot_series([X,Y],['X','Y'],save='series_x_y_SV.png',folder=img_dir,display=display_plots)
    # plot_series(X,'X',save='serie_x.png',folder=img_dir,display=display_plots)
    # plot_series(Y,'Y','tab:orange',save='serie_y.png',folder=img_dir,display=display_plots)
else:
    plot_series([X,Y],['X','Y'],display=display_plots)
    # plot_series(X,'X',display=display_plots)
    # plot_series(Y,'Y','tab:orange',display=display_plots)

if SIS:
    # Filtro de partículas
    filter = ParticleFilter1DStochasticVolatility(N,resample=False)

    start_SIS = time.time()
    filter.sv_params(SV_params)

    # sampleo de particulas iniciales}
    filter.set_initial_particles(0,(SIGMA**2)/(1-(ALPHA**2)))

    # aplicación manual de las primeras iteraciones
    # estos pasos están implementados en el método .step()
    filter.observe(Y[0])
    filter.sample_particles()
    filter.update_weights()
    filter._medias.append(np.average(filter.particle_sequence[-1],weights=filter.weights_sequence[-1]))

    # iterando sobre todas las observaciones

    for i in range(2,n_samples):
        filter.observe(Y[i])
        # actualizacion
        filter.step()
        if (i+1) % 100  == 0:
            print("Aprendizaje en linea {}%, transcurridos {}".format((i+1)/5,timeSince(start_SIS)))
    print("Aprendizaje en linea terminado luego de {}".format(timeSince(start_SIS)))

    X_pred_SIS = filter.medias
    mse_SIS = ((X - X_pred_SIS)**2).mean()
    print("Error cuadrático medio = {}".format(mse_SIS))

    # Visualización
    titulo = 'Filtro de particulas para SV'
    if save_plots:
        plot_series([X,X_pred_SIS],['X','X predicted'],save='particle_SIS.png',folder=img_dir,display=display_plots,title=titulo)
    else:
        plot_series([X,X_pred_SIS],['X','X predicted'],display=display_plots,title=titulo)


if SIR:
    # Filtro con resampling
    filterR = ParticleFilter1DStochasticVolatility(N,resample=True)

    start_SIR = time.time()

    filterR.sv_params(SV_params)

    # sampleo de particulas iniciales}
    filterR.set_initial_particles(0,(SIGMA**2)/(1-(ALPHA**2)))

    # iterando sobre todas las observaciones
    for i in range(1,n_samples):
        filterR.observe(Y[i])
        # actualizacion
        filterR.step()
        if (i+1) % 100  == 0:
            print("Aprendizaje en linea {}%, transcurridos {}".format((i+1)/5,timeSince(start_SIR)))
    print("Aprendizaje en linea terminado luego de {}".format(timeSince(start_SIR)))

    X_pred_SIR = filterR.medias
    mse_SIR = ((X - X_pred_SIR)**2).mean()
    print("Error cuadrático medio = {}".format(mse_SIR))

    # Visualización
    titulo = 'Filtro de particulas para SV con resampling'
    if save_plots:
        plot_series([X,X_pred_SIR],['X','X predicted'],save='particle_SIR.png',folder=img_dir,display=display_plots,title=titulo)
    else:
        plot_series([X,X_pred_SIR],['X','X predicted'],display=display_plots,title=titulo)
