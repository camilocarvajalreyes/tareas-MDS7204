import numpy as np
np.random.seed(0)
from plot import plot_posterior
from gp.gp_camilo import GaussianProcess
from kernels import SpectralMixtureKernel
from eval import eval

# Parámetros del código

np.random.seed(0)
entrenar = True
save_plots = True  # si guardar o no las figuras
img_dir = "unidad_2/tarea_3/informe/img"  # directorio en caso de guardar las figuras

# heart rates
hr1 = np.loadtxt('unidad_1/tarea_2/hr.7257.txt')

# eliminando parte de las observaciones
ALPHA = .3
indices = np.random.choice(len(hr1), int(len(hr1)*(1-ALPHA)), replace=False)
time = np.linspace(0,15,1800)
test_ind = np.array([i for i in range(len(hr1)) if i not in indices])

# titulo = "Puntos de entrenamiento y testeo, alpha = {}%".format((ALPHA*100))
# plot_series([hr1[indices],hr1[test_ind]],[times[indices],times[test_ind]],['Train','Test'],title=titulo)
# mas o menos equivalente a plot_data(gp)


gp = GaussianProcess()
gp.kernel = SpectralMixtureKernel()
gp.kernel.show_hypers()
print(f'\tsigma_n = {gp.sigma_n}')

gp.load(time[indices],hr1[indices])
# plot_data(gp)

gp.compute_posterior(where=time)

eval(test_ind,hr1[test_ind],gp,nombre_modelo='modelo sin entrenar')
titulo = "Posterior para GP sin entrenar, alpha={}%".format(ALPHA*100)
img_file = "untrained_gp_post.png" if save_plots else None
plot_posterior(gp,0, test_points=hr1[test_ind], test_times=time[test_ind],title=titulo,save=img_file,folder=img_dir)


# Entrenamiento
if entrenar:
    gp.train()
    titulo = "Posterior para GP entrenada, alpha={}%".format(ALPHA*100)
    gp.compute_posterior(where=time)

    print(f'Negative log-likelihood modelo entrenado: {gp.nll()}')
    eval(time[test_ind],hr1[test_ind],gp,nombre_modelo='modelo entrenado')
    # gp.kernel.show_hypers()
    img_file = "trained_gp_post.png" if save_plots else None
    plot_posterior(gp,0, test_points=hr1[test_ind],test_times=time[test_ind],save=img_file,folder=img_dir,title=titulo)
