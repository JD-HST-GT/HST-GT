import re
from sklearn.metrics import r2_score
def r2(pre,yt):
    pre = pre.view(-1).cpu().detach().numpy()
    yt = yt.view(-1).cpu().detach().numpy()

    sc = r2_score(pre,yt)
    return sc

from sklearn.metrics import mean_squared_error, mean_absolute_error
def mse(pre,yt):
    pre = pre.view(-1).cpu().detach().numpy()
    yt = yt.view(-1).cpu().detach().numpy()

    sc = mean_squared_error(pre,yt)
    return sc

def mae(pre,yt):
    pre = pre.view(-1).cpu().detach().numpy()
    yt = yt.view(-1).cpu().detach().numpy()

    sc = mean_absolute_error(pre,yt)
    return sc

def rmse(pre,yt):
    sc = mse(pre,yt) ** 0.5
    return sc


import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def mape(pre,yt):
    y_pred = pre.view(-1).cpu().detach().numpy()
    y_true = yt.view(-1).cpu().detach().numpy()

    return mean_absolute_percentage_error(pre,yt)

def smape(pre,yt):
    y_pred = pre.view(-1).cpu().detach().numpy()
    y_true = yt.view(-1).cpu().detach().numpy()
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def metric(pre,yt):
    res = {
        "r2_score":r2(pre,yt),
        "mae":mae(pre,yt),
        "mape":mape(pre,yt),
        "smape":smape(pre,yt),
        "mse":mse(pre,yt),
        "rmse":rmse(pre,yt)
    }
    return res


def metric_all(pres,yts):
    res = {}
    for i in pres:
        res[i] = metric(pres[i],yts[i])
        
    return res

