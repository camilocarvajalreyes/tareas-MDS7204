import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_series(series,time,y_tag,mode='scatter',colour='tab:blue',save=False,folder=None,display=True,title=None,fig_size=(18,5)):
    """
    Plotea una (o varias) serie(s) de tiempo

    Parametros
    ----------
        series: numpy.array o list(numpy.array)
            secuencia de valores o lista con varias secuencias de valores
        
        time: numpy.array o list(numpy.array)
            secuencia de valores o lista con tiempos de las realizaciones de serie

        obs: numpy.array o list(numpy.array)
            secuencia de indices de valores observados
        
        y_tag: str o list(str)
            etiqueta o etiquetas de las series a graficar

        colour: str 
            opcional, default 'tab:blue'. color (matplotlib) de las lineas de la series, solo valido cuando se plotea una sola serie

        save: str o bool
            opcional, default False. si es un string entonces guarda la figura con el nombre correspondiente
        
        folder: str
            opcional, default None. directorio en el cual se guarda la figura

        display: bool
            opcional, default True. si se muestran o no las figuras.
    
    """
    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=fig_size)
    if isinstance(y_tag,list):
        assert(len(series)==len(y_tag))
        for i in range(len(y_tag)):
            if mode=='line':
                sns.lineplot(x=time[i],y=series[i], ax=ax, markers=['o','o','o'])
            elif mode=='scatter':
                sns.scatterplot(x=time[i],y=series[i], ax=ax)
            else:
                raise ValueError("parámetro 'mode' debe ser o 'line' o bien 'scatter'")
        ax.legend(y_tag)
        if title is None:
            title = 'Realizaciones de series'
        ax.set(xlabel='tiempo',ylabel='valor',title=title)
    else:
        assert(len(series)==len(time))
        if mode=='line':
            sns.lineplot(x=time,y=series, ax=ax, markers=['o','o','o'], color=colour)
        elif mode=='scatter':
            sns.scatterplot(x=time,y=series, ax=ax)
        else:
            raise ValueError("parámetro 'mode' debe ser o 'line' o bien 'scatter'")
        if title is None:
            title = 'Realizaciones de la serie {}'.format(y_tag)
        ax.set(xlabel='tiempo',ylabel='valor de '+y_tag,title=title)
    if display:
        plt.show()
    if save:
        path = folder+ '/' + save
        plt.savefig(path)


def plot_posterior(gp_obj,n_samples=0,test_points=None,test_times=None,fig_size=(18,5),title=None,save=False,folder=None):
    plt.figure(figsize=fig_size)
    plt.plot(gp_obj.time,gp_obj.mean, 'tab:purple', label='posterior')

    plt.plot(gp_obj.x,gp_obj.y, '.b', markersize = 8, label='data')

    if test_points is not None:
        plt.scatter(x=test_times,y=test_points, color='tab:orange', label='test data')

    error_bars = 2 * np.sqrt((np.diag(gp_obj.cov)))
    plt.fill_between(gp_obj.time, gp_obj.mean - error_bars, gp_obj.mean + error_bars, color='blue',alpha=0.1, label='95% error bars')
    if n_samples > 0:
        gp_obj.compute_posterior(where = gp_obj.time)
        gp_obj.sample(how_many = n_samples)
        plt.plot(gp_obj.time,gp_obj.samples,alpha = 0.7)
    if title is None:
        plt.title('Posterior')
    else:
        plt.title(title)
    plt.xlabel('time')
    plt.legend(loc=1, ncol=3)
    plt.xlim([min(gp_obj.time),max(gp_obj.time)])
    # plt.ylim([-v_axis_lims,v_axis_lims])
    plt.tight_layout()

    if save:
        path = folder+ '/' + save
        plt.savefig(path)
    else:
        plt.show()


def plot_data(gp_obj,fig_size=(18,5),title=None,save=False,folder=None):
    plt.figure(figsize=fig_size)

    plt.plot(gp_obj.x,gp_obj.y, '.r', markersize = 8,label='data')

    if title is None:
        plt.title('Posterior')
    else:
        plt.title(title)
    
    plt.xlabel('time')
    plt.legend(loc=1)
    plt.xlim([min(gp_obj.time),max(gp_obj.time)])
    # plt.ylim([-v_axis_lims,v_axis_lims])
    plt.tight_layout()
    
    if save:
        path = folder+ '/' + save
        plt.savefig(path)
    else:
        plt.show()
