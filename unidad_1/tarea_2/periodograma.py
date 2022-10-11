from scipy.signal import periodogram
import numpy as np
from plot import plot_series, plot_spectrum

img_dir = "unidad_1/tarea_2/informe/img"

# ejemplo cosenos
N = int(2**10)

omega = 10*2*np.pi
cosenos = lambda t: np.sum([n*np.cos(n*omega*t) for n in range(1,6)])
t_max = 0.5

serie_cos = np.array([cosenos(t) for t in np.linspace(0,t_max,N)])

plot_series(serie_cos,'suma de cosenos',fig_size=(8,5),display=False,save='serie_cos.png',folder=img_dir)

f, Pxx_den = periodogram(serie_cos)
plot_spectrum(f,Pxx_den,'suma de cosenos',log=False,fig_size=(8,5),max_freq=0.1,colour='tab:purple',display=False,save='psd_cos.png',folder=img_dir)

# heart rates
hr1 = np.loadtxt('unidad_1/tarea_2/hr.7257.txt')
# plot_series(hr1,'Pulso sanguineo 1')
hr2 = np.loadtxt('unidad_1/tarea_2/hr.11839.txt')
# plot_series(hr2,'Pulso sanguineo 2')
plot_series([hr1,hr2],['pulso 1','pulso 2'],title='Series de pulso sanguineo',fig_size=(8,5),display=False,save='serie_hr.png',folder=img_dir)

f1, Pxx_den1 = periodogram(hr1)
# plot_spectrum(f1,Pxx_den1,'pulso 1',log=False)
f2, Pxx_den2 = periodogram(hr2)
# plot_spectrum(f2,Pxx_den2,'pulso 2',log=False)

# print('Forma de f (sample frequencies): {}'.format(f1.shape))
# print('Forma de Pxx_den (power spectral density): {}'.format(Pxx_den1.shape))

plot_spectrum(f1,[Pxx_den1,Pxx_den2],['pulso 1','pulso 2'],max_freq=0.1,fig_size=(8,5),display=False,save='psd_hr.png',folder=img_dir)
