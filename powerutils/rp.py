import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator

from powerutils.common import check_timeseries, check_timeframe

COEFFS = np.array([
    [0.9738, 0.0262],
    [0.9155, 0.0845],
    [0.8572, 0.1428],
    [0.7989, 0.2011],
    [0.7406, 0.2594],
    [0.6823, 0.3177],
    [0.6240, 0.3760],
    [0.5657, 0.4343],
    [0.5074, 0.4926],
    [0.4491, 0.5509],
    [0.3908, 0.6092],
    [0.3325, 0.6675],
    [0.2742, 0.7258],
    [0.2159, 0.7841],
    [0.1576, 0.8424],
    [0.0993, 0.9007]
])

class RpModel(BaseEstimator):
    '''
    RP模型.

    fixed_coeffs为True时使用固定系数, 并使fit方法无效.
    fixed_coeffs为False时, 从数据中拟合出系数.
    coeffs_属性为(16, 2)的拟合系数.
    '''
    def __init__(self, fixed_coeffs=False):
        self.fixed_coeffs = fixed_coeffs

    @staticmethod
    def _fit_coeff(Xi, yi):
        '''
        Xi (n, 2), 第一列是t=-1的实际功率, 第二列是t=i的短期预测功率.
        yi (n,), t=i的实际功率. 要求Xi和yi都不含缺测.
        '''
        eps = 1e-10
        fun = lambda x: np.abs((x * Xi).sum(axis=1) - yi).sum()
        constraints = (
            {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1},
            {'type': 'ineq', 'fun': lambda x: x[0] - eps},
            {'type': 'ineq', 'fun': lambda x: x[1] - eps}
        )
        x0 = np.array([0.5, 0.5])
        res = minimize(fun, x0, method='SLSQP', constraints=constraints)
        coeff = res.x

        return coeff

    def fit(self, X=None, y=None):
        '''
        要求X的列名形如[实际功率(t=i), 短期预测功率(t=i)]
        要求y的列名形如[实际功率(t=i)]
        '''
        if self.fixed_coeffs:
            self.coeffs_ = COEFFS
        else:
            check_timeframe(X, y)
            self.coeffs_ = np.zeros((16, 2))
            for i in range(16):
                Xi = X[['实际功率(t=-1)', f'短期预测功率(t={i})']]
                yi = y[f'实际功率(t={i})']
                self.coeffs_[i, :] = self._fit_coeff(Xi, yi)

        return self

    def predict(self, X):
        '''预测未来16步的功率. 允许X含缺测.'''
        data = {}
        for i in range(16):
            Xi = X[['实际功率(t=-1)', f'短期预测功率(t={i})']]
            # 列名取实际功率, 方便与y进行比较.
            data[f'实际功率(t={i})'] = (
                (self.coeffs_[i, :] * Xi).sum(axis=1, skipna=False)
            )
        yp = pd.DataFrame(data)

        return yp

def calc_step(y_true, y_short, step):
    '''用实际功率和短期预测功率计算第step步的超短期预测功率.'''
    check_timeseries(y_true, y_short)
    if step < 1 or step > 16:
        raise ValueError('step在1-16之间')

    a, b = COEFFS[step - 1, :]
    y_true = y_true.shift(step, freq='15T')
    y_ultra = a * y_true + b * y_short
    y_ultra.name = f'ultra-{step}'

    return y_ultra