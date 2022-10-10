from FourierTransform import DiscreteFourierTransform, FastFourierTransform
from plot import plot_times
import numpy as np
np.random.seed(0)

import time

def timeSince(since):
    now = time.time()
    return now - since

N = int(2**10)

omega = 10*2*np.pi
cosenos = lambda t: np.sum([n*np.cos(n*omega*t) for n in range(1,6)])
t_max = 0.5

# series de ejemplo
serie_cos = np.array([cosenos(t) for t in np.linspace(0,t_max,N)])
serie_cos_ruido = np.array([cosenos(t)+np.random.normal(0,0.5) for t in np.linspace(0,t_max,N)])
hr1 = np.loadtxt('unidad_1/tarea_2/hr.7257.txt')[:N]
hr2 = np.loadtxt('unidad_1/tarea_2/hr.11839.txt')[:N]

t_cos, t_ruido =  {'dft':np.zeros(10), 'fft':np.zeros(10)}, {'dft':np.zeros(10), 'fft':np.zeros(10)}
t_hr1, t_hr2 = {'dft':np.zeros(10), 'fft':np.zeros(10)}, {'dft':np.zeros(10), 'fft':np.zeros(10)}

for i in range(10):
    for t, serie in zip([t_cos, t_ruido, t_hr1, t_hr2],[serie_cos,serie_cos_ruido,hr1,hr2]):
        t_init = time.time()
        DiscreteFourierTransform(serie[:int(2**(i+1))])
        t['dft'][i] = timeSince(t_init)
        t_init = time.time()
        FastFourierTransform(serie[:int(2**(i+1))])
        t['fft'][i] = timeSince(t_init)

labs = ['promedio dft', 'promedio fft', '$M_1 N^2$','$M_2 N \log(N)$','coseno dft', 'coseno fft', 
        'coseno ruido dft', 'coseno ruido fft','hr1 dft', 'hr1 fft', 'hr2 dft', 'hr2 fft']
avg_dft = np.average([t_cos['dft'], t_ruido['dft'], t_hr1['dft'], t_hr2['dft']],axis=0)
avg_fft = np.average([t_cos['fft'], t_ruido['fft'], t_hr1['fft'], t_hr2['fft']],axis=0)

orden_n2 = np.array([(1/22048576)*(2**n)**2 for n in range(1,11)])
orden_log = np.array([(1/360000)*(2**n)*np.log(2**n) for n in range(1,11)])
tiempos = [avg_dft, avg_fft, orden_n2, orden_log, t_cos['dft'], t_cos['fft'], t_ruido['dft'], t_ruido['fft'], 
            t_hr1['dft'], t_hr1['fft'], t_hr2['dft'], t_hr2['fft']]

img_dir = "unidad_1/tarea_2/informe/img"
plot_times(np.array([2**i for i in range(1,11)]),tiempos,labs,mk=4,save='complejidad_fft1.png',folder=img_dir,display=False)

N = int(2**15)
serie_cos_ruido2 = np.array([cosenos(t)+np.random.normal(0,0.5) for t in np.linspace(0,t_max,N)])

t_ruido2 =  {'dft':np.zeros(15), 'fft':np.zeros(15)}

for i in range(15):
    t_init = time.time()
    DiscreteFourierTransform(serie_cos_ruido2[:int(2**(i+1))])
    t_ruido2['dft'][i] = timeSince(t_init)
    t_init = time.time()
    FastFourierTransform(serie_cos_ruido2[:int(2**(i+1))])
    t_ruido2['fft'][i] = timeSince(t_init)

labs2 = ['coseno ruido dft', 'coseno ruido fft', '$M_1 N^2$','$M_2 N \log(N)$']

orden_n2 = np.array([(1/22048576)*(2**n)**2 for n in range(1,16)])
orden_log = np.array([(1/360000)*(2**n)*np.log(2**n) for n in range(1,16)])
tiempos2 = [t_ruido2['dft'], t_ruido2['fft'], orden_n2, orden_log]

plot_times(np.array([2**i for i in range(1,16)]),tiempos2,labs2,log=True,mk=4,save='complejidad_fft2.png',folder=img_dir,display=False)
