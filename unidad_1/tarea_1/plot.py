import seaborn as sns
import matplotlib.pyplot as plt

def plot_series(series,y_tag,colour='tab:blue'):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12,4))
    if isinstance(y_tag,list):
        assert(len(series)==len(y_tag))
        for i in range(len(y_tag)):
            sns.lineplot(data=series[i], ax=ax, markers=['o','o','o'])
        ax.legend(y_tag)
        ax.set(xlabel='tiempo',ylabel='valor',title='Realizaciones de series')
    else:
        sns.lineplot(data=series, ax=ax, markers=['o','o','o'], color=colour)
        ax.set(xlabel='tiempo',ylabel='valor de '+y_tag,title='Realizaciones de la serie {}'.format(y_tag))
    plt.show()
