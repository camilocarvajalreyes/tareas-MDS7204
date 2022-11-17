import numpy as np
np.random.seed(0)
from plot import plot_posterior
from gp.gp_lite_tarea import gp_lite

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
test_ind = np.array([i for i in range(len(hr1)) if i not in indices])

# titulo = "Puntos de entrenamiento y testeo, alpha = {}%".format((ALPHA*100))
# plot_series([hr1[indices],hr1[test_ind]],[times[indices],times[test_ind]],['Train','Test'],title=titulo)
# mas o menos equivalente a plot_data(gp)


gp = gp_lite()
gp.init_hypers()
gp.show_hypers()


# gp.sample(how_many=2)
# gp.plot_samples()

gp.load(indices,hr1[indices])
# plot_data(gp)

gp.compute_posterior(where=np.arange(len(hr1)))

print(f'negative log-likelihood modelo sin entrenar: {gp.nll()}')
titulo = "Posterior para GP sin entrenar, alpha={}%".format(ALPHA*100)
img_file = "untrained_gp_post.png" if save_plots else None
plot_posterior(gp,0, test_points=hr1[test_ind], test_times=test_ind,title=titulo,save=img_file,folder=img_dir)


# Entrenamiento
if entrenar:
    gp.train()
    titulo = "Posterior para GP entrenada, alpha={}%".format(ALPHA*100)
    gp.compute_posterior(where=np.arange(len(hr1)))

    print(f'Negative log-likelihood modelo entrenado: {gp.nll()}')
    gp.show_hypers()
    img_file = "trained_gp_post.png" if save_plots else None
    plot_posterior(gp,0, test_points=hr1[test_ind],test_times=test_ind,save=img_file,folder=img_dir,title=titulo)
