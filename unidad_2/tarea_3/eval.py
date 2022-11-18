from scipy.stats import multivariate_normal

def MSE(times,y_true,gp_obj):
    means, _ = gp_obj.pred_dist(where=times)
    return ((y_true - means)**2).mean()

def log_likelihood(times,y_true,gp_obj):
    means, covs = gp_obj.pred_dist(where=times)
    return multivariate_normal.logpdf(y_true,means,covs)

def eval(times,y_true,gp_obj,nombre_modelo=''):
    print("Log-verosimilitud negativa conjunto train {}: {}".format(nombre_modelo,gp_obj.nll()))
    print("Log-verosimilitud negativa conjunto test {}: {}".format(nombre_modelo,gp_obj.nll(times,y_true)))
    # print("Log-verosimilitud negativa conjunto train {}: {}".format(nombre_modelo,-1*log_likelihood(gp_obj.x,gp_obj.y,gp_obj)))
    # print("Log-verosimilitud negativa conjunto test {}: {}".format(nombre_modelo,-1*log_likelihood(times,y_true,gp_obj)))
    print("Error cuadratico medio conjunto train {}: {}".format(nombre_modelo,MSE(gp_obj.x,gp_obj.y,gp_obj)))
    print("Error cuadratico medio conjunto test {}: {}".format(nombre_modelo,MSE(times,y_true,gp_obj)))
