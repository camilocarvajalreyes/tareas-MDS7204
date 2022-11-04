import numpy as np
np.random.seed(0)
from plot import plot_series, plot_data, plot_posterior
from gp.gp_lite_tarea import gp_lite

# heart rates
hr1 = np.loadtxt('unidad_1/tarea_2/hr.7257.txt')
times = np.linspace(0,100,len(hr1))
# hr2 = np.loadtxt('unidad_1/tarea_2/hr.11839.txt')

# eliminando parte de las observaciones
ALPHA = .3
indices = np.random.choice(len(hr1), int(len(hr1)*(1-ALPHA)), replace=False)

# plot_series(hr1,times,'Pulso sanguineo 1',obs=indices,obs_only=True)
# equivalente a plot_data(gp)


gp = gp_lite()
gp.init_hypers()

i = len(indices)
gp.compute_posterior(dimension=i)

gp.sample(how_many=2)
# gp.plot_samples()

gp.load(times[indices],hr1[indices])
plot_data(gp)

gp.compute_posterior(dimension=1000)
# gp.plot_posterior(5,v_axis_lims = 35)
plot_posterior(gp,5)
print(f'negative log-likelihood: {gp.nll()}')
