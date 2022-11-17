from scipy.stats import multivariate_normal

def MSE(times,y_true,gp_obj):
    means, _ = gp_obj.pred_dist(where=times)
    return ((y_true - means)**2).mean()

def log_likelihood(times,y_true,gp_obj):
    means, covs = gp_obj.pred_dist(where=times)
    return multivariate_normal.logpdf(y_true,means,covs)

def eval(times,y_true,gp_obj,nombre_modelo=''):
    print("Error cuadratico medio conjunto test {}: {}".format(nombre_modelo,MSE(times,y_true,gp_obj)))
    print("Log-likelihood conjunto test {}: {}".format(nombre_modelo,log_likelihood(times,y_true,gp_obj)))
