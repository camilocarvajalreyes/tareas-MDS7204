from scipy.stats import multivariate_normal

def predict(times,gp_obj):
    means, _ = gp_obj.pred_dist(where=times)
    return means

def MSE(y_pred,y_true):
    return ((y_true - y_pred)**2).mean()

def log_likelihood(times,y_true,gp_obj):
    means, covs = gp_obj.pred_dist(where=times)
    return multivariate_normal.logpdf(y_true,means,covs)

def eval(times,y_true,gp_obj,nombre_modelo=''):
    y_pred = predict(times,gp_obj)
    y_pred_train = predict(gp_obj.x,gp_obj)
    print("Log-verosimilitud negativa conjunto train {}: {}".format(nombre_modelo,gp_obj.nll()))
    # print("Log-verosimilitud negativa conjunto test {}: {}".format(nombre_modelo,gp_obj.nll(times,y_true)))
    # print("Log-verosimilitud negativa conjunto train {}: {}".format(nombre_modelo,-1*log_likelihood(gp_obj.x,gp_obj.y,gp_obj)))
    print("Log-verosimilitud negativa conjunto test {}: {}".format(nombre_modelo,-1*log_likelihood(times,y_true,gp_obj)))
    print("Error cuadratico medio conjunto train {}: {}".format(nombre_modelo,MSE(y_pred_train,gp_obj.y)))
    print("Error cuadratico medio conjunto test {}: {}".format(nombre_modelo,MSE(y_pred,y_true)))
