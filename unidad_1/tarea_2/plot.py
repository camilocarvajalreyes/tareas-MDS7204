from typing import Type
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_series(series,y_tag,colour='tab:blue',save=False,folder=None,display=True,title=None):
    """
    Plotea una (o varias) serie(s) de tiempo

    Parametros
    ----------
        series: numpy.array o list(numpy.array)
            secuencia de valores o lista con varias secuencias de valores

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
    _, ax = plt.subplots(figsize=(12,5))
    if isinstance(y_tag,list):
        assert(len(series)==len(y_tag))
        for i in range(len(y_tag)):
            sns.lineplot(data=series[i], ax=ax, markers=['o','o','o'])
        ax.legend(y_tag)
        if title is None:
            title = 'Realizaciones de series'
        ax.set(xlabel='tiempo',ylabel='valor',title=title)
    else:
        sns.lineplot(data=series, ax=ax, markers=['o','o','o'], color=colour)
        if title is None:
            title = 'Realizaciones de la serie {}'.format(y_tag)
        ax.set(xlabel='tiempo',ylabel='valor de '+y_tag,title=title)
    if display:
        plt.show()
    if save:
        path = folder+ '/' + save
        plt.savefig(path)


def plot_spectrum(frequencies,PSD,y_tag,log=False,max_freq=None,colour='tab:blue',save=False,folder=None,display=True,title=None):
    sns.set_theme(style="whitegrid")
    _, ax = plt.subplots(figsize=(12,5))
    if isinstance(y_tag,list):
        assert(len(PSD)==len(y_tag))
        for i in range(len(y_tag)):
            if log:
                ax.semilogy(frequencies,PSD[i])
            else:
                ax.plot(frequencies,PSD[i])
        ax.legend(y_tag)
        if title is None:
            title = 'Espectros de series'
        ax.set(xlabel='frecuencia',ylabel='potencia',title=title)
    else:
        if log:
            ax.semilogy(frequencies,PSD,c=colour)
        else:
            ax.plot(frequencies,PSD,c=colour)
        if title is None:
            title = 'Espectro de la serie {}'.format(y_tag)
            ax.set(xlabel='frecuencia',ylabel='potencia',title=title)
    if max_freq is not None:
        ax.set_xlim(left=0,right=max_freq)
    if display:
        plt.show()
    if save:
        path = folder+ '/' + save
        plt.savefig(path)
