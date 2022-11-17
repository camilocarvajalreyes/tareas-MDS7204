def MSE(times,y_true,gp_obj):
    means, _ = gp_obj.pred_dist(where=times)
    return ((y_true - means)**2).mean()
