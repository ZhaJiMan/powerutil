import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_validate

def mean_error(y_true, y_pred, axis=None):
    '''计算平均误差. 默认聚合为标量.'''
    return np.asarray(y_pred - y_true).mean(axis=axis)

class DataFrameScaler(BaseEstimator, TransformerMixin):
    '''对DataFrame做缩放的类, 会保留DataFrame的索引信息.'''
    def __init__(self, scaler):
        self.scaler = scaler

    @staticmethod
    def _check_dataframe(X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X应该是DataFrame')

    def fit(self, X, y=None):
        self._check_dataframe(X)
        self.scaler.fit(X.to_numpy())

        return self

    def transform(self, X):
        self._check_dataframe(X)
        data = self.scaler.transform(X.to_numpy())
        X = pd.DataFrame(data, index=X.index, columns=X.columns)

        return X

class TimeSeriesSplit2:
    '''将时间序列按顺序划分成n_splits组训练-测试集, 并且每组的样本量大致相同.'''
    def __init__(self, n_splits=2):
        if n_splits < 2:
            raise ValueError('n_splits至少为2')
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        size, rest = divmod(len(X), self.n_splits + 1)
        if size == 0:
            raise ValueError('样本量不足以分组')

        for i in range(self.n_splits):
            step1 = size + 1 if i < rest else size
            start1 = i * step1
            stop1 = start1 + step1
            train = list(range(start1, stop1))

            step2 = size + 1 if i + 1 < rest else size
            start2 = stop1
            stop2 = start2 + step2
            test = list(range(start2, stop2))
            yield train, test

def cross_val_table(estimator, X, y=None, scoring=None, cv=None, n_jobs=None):
    '''交叉验证模型, 以DataFrame的形式返回每个子集上的评分.'''
    cv_results = cross_validate(
        estimator, X=X, y=y, scoring=scoring, cv=cv,
        n_jobs=n_jobs, error_score='raise'
    )
    table = pd.DataFrame(cv_results).filter(like='test_')
    table.index = [f'cv{i + 1}' for i in range(table.shape[0])]
    table.columns = table.columns.str[5:]
    table.loc['mean'] = table.mean()

    return table