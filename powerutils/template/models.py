import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt

from powerutils.mlutils import mean_error
from powerutils.common import pandas_like
from powerutils.accuracy import *

class Baseline(BaseEstimator):
    '''用文件结果作为预测结果.'''
    def __init__(self, label):
        self.label = label

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        return X[self.label]

class Corrector(BaseEstimator, TransformerMixin):
    '''对特征进行订正的模型.'''
    def __init__(
        self, regressor, features, label):
        self.regressor = regressor
        self.features = features
        self.label = label

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X[self.features])
        y = X[self.label]
        self.regressor.fit(X, y)

        return self

    def predict(self, X):
        Xt = self.scaler_.transform(X[self.features])
        yp = self.regressor.predict(Xt)
        yp = pd.Series(yp, index=X.index)

        return yp

    def transform(self, X):
        return X.assign(**{self.label: self.predict(X)})

    def score(self, X, y=None):
        y_true = X[self.label]
        y_pred = self.predict(X)
        metrics = {
            'BIAS': mean_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

        return metrics

class Predictor(BaseEstimator):
    '''短期预测模型.'''
    def __init__(self, regressor, features):
        self.regressor = regressor
        self.features = features

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        Xt = self.scaler_.fit_transform(X[self.features])
        self.regressor.fit(Xt, y)

        return self

    def predict(self, X):
        Xt = self.scaler_.transform(X[self.features])
        yp = self.regressor.predict(Xt)
        yp = pd.Series(yp, index=X.index)

        return yp

class Plotter:
    '''画出模型预测结果的类.'''
    def __init__(self, cap):
        self.cap = cap
        self.models = {}

    def add_model(self, model, name, color):
        '''添加一个模型.'''
        self.models[name] = {'model': model, 'color': color}

    def del_model(self, name):
        '''删除一个模型.'''
        del self.models[name]

    def fit(self, X, y, cv=5, n_jobs=None):
        '''缓存每个模型cross_val_predict的结果.'''
        for attrs in self.models.values():
            yp = cross_val_predict(attrs['model'], X, y, cv=cv, n_jobs=n_jobs)
            attrs['pred'] = pandas_like(yp, y).asfreq('15T')
        self.y_true = y.asfreq('15T')

    def plot_pred(self, start_time=None, end_time=None):
        '''画两张子图, 第一张是各模型功率曲线, 第二张是各模型预测误差.'''
        slicer = slice(start_time, end_time)
        y_true = self.y_true.loc[slicer]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.subplots_adjust(hspace=0.15)
        y_true.plot(ax=axes[0], lw=1, c='dimgray', label='实际功率')

        strings = []
        for name, attrs in self.models.items():
            y_pred = attrs['pred'].loc[slicer]
            y_pred.plot(ax=axes[0], lw=1, c=attrs['color'], label=name)
            err = southern_error(y_true, y_pred, self.cap)
            err.plot(ax=axes[1], lw=1, c=attrs['color'], label=name)
            acc1 = national_accuracy(y_true, y_pred, self.cap, positive=True).mean()
            acc2 = southern_accuracy(y_true, y_pred, self.cap).mean()
            string = f'{name}: 国网准确率={acc1:.2f}%, 南网准确率={acc2:.2f}%'
            strings.append(string)

        axes[0].axhline(0.1 * self.cap, c='k', lw=1, ls='--')
        axes[0].set_ylabel('功率(MW)', fontsize='medium')
        axes[1].set_ylabel('预测误差', fontsize='medium')

        info = '\n'.join(strings)
        axes[0].set_title('预测功率', loc='left', fontsize='medium')
        axes[0].set_title(info, loc='right', fontsize='medium')
        axes[1].set_title('预测误差', loc='left', fontsize='medium')

        for ax in axes:
            ax.legend(loc='upper right')
            ax.set_xmargin(0)
            ax.set_ylim(0, None)
            plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
            ax.set_xlabel('')

        return axes

    def plot_acc(self, start_time=None, end_time=None):
        '''画出各模型的日准确率曲线.'''
        slicer = slice(start_time, end_time)
        y_true = self.y_true.loc[slicer]

        _, ax = plt.subplots(figsize=(10, 4))
        for name, attrs in self.models.items():
            y_pred = attrs['pred'].loc[slicer]
            acc = southern_accuracy(y_true, y_pred, self.cap)
            acc.plot(ax=ax, lw=1, c=attrs['color'], label=name)
        ax.legend(loc='lower right')
        ax.axhline(60, lw=1, c='gray', ls='--')
        ax.set_ylim(None, 100)

        ax.set_ylabel('准确率(%)', fontsize='medium')
        ax.set_title('日准确率', loc='left', fontsize='medium')

        return ax