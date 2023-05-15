'''不同省的准确率计算公式各有不同, 建议单独实现.'''
import numpy as np
import pandas as pd

from powerutils.common import check_timeseries

def national_error(y_true, y_pred, cap, positive=False):
    '''国网准确率公式在每个时刻的误差. positive指定是否只考虑实际功率大于零的时刻.'''
    check_timeseries(y_true, y_pred)
    err = (y_true - y_pred)**2 / cap**2
    if positive:
        err.where(y_true > 0, inplace=True)

    return err

def national_accuracy(y_true, y_pred, cap, positive=False):
    '''计算每日的国网准确率. positive指定是否只考虑实际功率大于零的时刻.'''
    err = national_error(y_true, y_pred, cap, positive)
    err = err.resample('D').mean()
    acc = (1 - np.sqrt(err)) * 100

    return acc

def southern_error(y_true, y_pred, cap):
    '''南网准确率公式在每个时刻的误差.'''
    check_timeseries(y_true, y_pred)
    y_true, y_pred = y_true.align(y_pred)

    c1 = 0.1 * cap
    c2 = 0.2 * cap
    cond = (y_true < c1) & (y_pred < c1)
    dy = (y_true - y_pred).mask(cond)
    err = np.where(y_true < c2, dy / c2, dy / y_true)**2
    err = pd.Series(err, index=y_true.index)

    return err

def southern_accuracy(y_true, y_pred, cap):
    '''计算每日的南网准确率.'''
    err = southern_error(y_true, y_pred, cap)
    err = err.resample('D').mean()
    acc = (1 - np.sqrt(err)) * 100

    return acc

def northern_error(y_true, y_pred, cap):
    '''华北电网准确率公式在每个时刻的误差.'''
    check_timeseries(y_true, y_pred)
    dy = (y_true - y_pred).abs()
    # 将误差全为NaN或0的日子的weight设为1.
    weight = dy.resample('D').transform('sum').where(lambda x: x > 0, 1)
    err = dy**2 * dy / weight / cap**2

    return err

def northern_accuracy(y_true, y_pred, cap):
    '''计算每日的华北电网准确率.'''
    err = northern_error(y_true, y_pred, cap)
    err = err.resample('D').sum(min_count=1)
    acc = (1 - np.sqrt(err)) * 100

    return acc

def is_over_level(acc, level):
    '''判断日准确率是否高于level. 0和1表示是否, 含NaN.'''
    return (acc > level).astype(float).mask(acc.isna())