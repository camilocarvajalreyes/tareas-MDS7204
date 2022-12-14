import numpy as np
from plot import plot_series, plot_heatmaps
from filtros.filtro_kalman import KalmanFilter1D

# parámetros de este código
np.random.seed(0)
save_plots = True  # si guardar o no las figuras
img_dir = "unidad_1/tarea_1/informe/img"  # directorio en caso de guardar las figuras
display_plots = False  # si se muestran o no las figuras (pop-up), no funciona al mismo tiempo que save_plots

# Simulación de las series

print("Sampleando los parámetros")
A = np.random.uniform(0,1)
B = np.random.uniform(0.8,1.2)
C = np.random.uniform(0,1)
D = np.random.uniform(0.3,0.7)
X0 = np.random.normal(0,0.8)

print("Valor de A = {}\nValor de B = {}\nValor de C = {}\nValor de D = {}\nPosición inicial = {}".format(A,B,C,D,X0))

n_samples = 200

V = np.random.normal(0,1,n_samples)
W = np.random.normal(0,1,n_samples)

X = np.zeros(n_samples)
Y = np.zeros(n_samples)
X[0] = X0
for i in range(n_samples):
    try:
        X[i+1] = A*X[i] + B * V[i]
    except IndexError:
        pass
    Y[i] = C*X[i] + D*W[i]

if save_plots:
    plot_series([X,Y],['X','Y'],save='series_x_y.png',folder=img_dir,display=display_plots)
    # plot_series(X,'X',save='serie_x.png',folder=img_dir,display=display_plots)
    # plot_series(Y,'Y','tab:orange',save='serie_y.png',folder=img_dir,display=display_plots)
else:
    plot_series([X,Y],['X','Y'],display=display_plots)
    # plot_series(X,'X',display=display_plots)
    # plot_series(Y,'Y','tab:orange',display=display_plots)

# Filtro de Kalman

# definiendo el filtro
filter = KalmanFilter1D(Y[0],0.8,A,B,C,D)

# aplicación manual de las primeras iteraciones
filter.predict()
filter.x_predicted()
filter.observe(Y[1])
# print(filter._x_observed)
filter.update()
# print(filter.x_updated())

# iterando sobre todas las observaciones

for i in range(2,n_samples):
    # predicción
    filter.predict()
    # observación
    filter.observe(Y[i])
    # actualizacion
    filter.update()

# Visualización
X_pred = filter.x_predicted()
# plot_series(X_pred,'X predicted')

X_updt = filter.x_updated()
# plot_series(X_updt,'X updated')

# mse
mse = ((X - X_pred)**2).mean()
mse_updt = ((X - X_updt)**2).mean()
print("Error cuadrático medio pre-actualización = {}".format(mse))
print("Error cuadrático medio post-actualización = {}".format(mse_updt))

if save_plots:
    plot_series([X,X_updt],['X','X updated'],save='x_x_updt.png',folder=img_dir,display=display_plots)
    plot_series([X,X_pred,X_updt],['X','X predicted','X updated'],save='x_x_updt_x_pred.png',folder=img_dir,display=display_plots)
else:
    plot_series([X,X_updt],['X','X updated'],display=display_plots)
    plot_series([X,X_pred,X_updt],['X','X predicted','X updated'],display=display_plots)

def eval_variances(V:list,W:list,X,Y):
    # mse_dict_pred, mse_dict_updt = {}, {}
    mse_dict_pred, mse_dict_updt = np.zeros((len(V),len (W))), np.zeros((len(V),len (W)))
    for l, v in enumerate(V):
        # mse_dict_pred[v], mse_dict_updt[v] = {}, {}
        for j, w in enumerate(W):
            filter = KalmanFilter1D(Y[0],0.8,A,v,C,w)
            # filter.predict()
            for i in range(1,n_samples):
                filter.predict()
                filter.observe(Y[i])
                filter.update()

            mse_dict_pred[l][j] = ((X - filter.x_predicted())**2).mean()
            mse_dict_updt[l][j] = ((X - filter.x_updated())**2).mean()

    return mse_dict_pred, mse_dict_updt

Vs = [2.0, 1.2, 0.8, 0.4, 0.01]
Ws = [0.01, 0.15, 0.3, 0.7, 1.0]
mse_dict_pred, mse_dict_updt = eval_variances(Vs,Ws,X,Y)

# print("Valores de MSE para distintas varianzas, pre-actualización")
# print(mse_dict_pred)  # esto fue medio idiota pues no se usa la info de las varianzas
print("Valores de MSE para distintas varianzas, post-actualización")
print(mse_dict_updt)
# plot_heatmaps([mse_dict_pred,mse_dict_updt],Vs,Ws)
if save_plots:
    plot_heatmaps([mse_dict_updt],Vs,Ws,title='MSE versus varianzas V y W',display=False,save='variance_heatmap.png',folder=img_dir)
if display_plots:
    plot_heatmaps([mse_dict_updt],Vs,Ws,title='MSE versus varianzas V y W')
