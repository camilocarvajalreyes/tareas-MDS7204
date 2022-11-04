import numpy as np
np.random.seed(0)
from plot import plot_series

# heart rates
hr1 = np.loadtxt('unidad_1/tarea_2/hr.7257.txt')
# plot_series(hr1,'Pulso sanguineo 1')
# hr2 = np.loadtxt('unidad_1/tarea_2/hr.11839.txt')
# plot_series(hr2,'Pulso sanguineo 2')

# eliminando parte de las observaciones
ALPHA = .2
indices = np.random.choice(len(hr1), int(len(hr1)*(1-ALPHA)), replace=False)

# plot_series([hr1,hr2],['pulso 1','pulso 2'],obs=[indices,indices],title='Series de pulso sanguineo',fig_size=(8,5),display=True)
plot_series(hr1,'Pulso sanguineo 1',obs=indices, obs_only=True)
