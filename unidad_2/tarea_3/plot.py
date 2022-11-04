import seaborn as sns
import matplotlib.pyplot as plt


def plot_series(series,y_tag,obs=None,obs_only=False,colour='tab:blue',save=False,folder=None,display=True,title=None,fig_size=(18,5)):
    """
    Plotea una (o varias) serie(s) de tiempo

    Parametros
    ----------
        series: numpy.array o list(numpy.array)
            secuencia de valores o lista con varias secuencias de valores

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
            if not obs_only:
                sns.lineplot(data=series[i], ax=ax, markers=['o','o','o'])
            if obs is not None:
                sns.scatterplot(data=series[i][obs[i]], ax=ax)
        ax.legend(y_tag)
        if title is None:
            title = 'Realizaciones de series'
        ax.set(xlabel='tiempo',ylabel='valor',title=title)
    else:
        if not obs_only:
            sns.lineplot(data=series, ax=ax, markers=['o','o','o'], color=colour)
        if obs is not None:
            sns.scatterplot(x=obs,y=series[obs], ax=ax, color='tab:red')
        if title is None:
            title = 'Realizaciones de la serie {}'.format(y_tag)
        ax.set(xlabel='tiempo',ylabel='valor de '+y_tag,title=title)
    if display:
        plt.show()
    if save:
        path = folder+ '/' + save
        plt.savefig(path)
