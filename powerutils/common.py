import shutil
from pathlib import Path
from collections.abc import Iterable

import pandas as pd

def new_dir(dirpath):
    '''新建目录.'''
    dirpath = Path(dirpath)
    if not dirpath.exists():
        dirpath.mkdir(parents=True)

def renew_dir(dirpath):
    '''重建目录.'''
    dirpath = Path(dirpath)
    if dirpath.exists():
        shutil.rmtree(dirpath)
    new_dir(dirpath)

def prefix_dict(d, prefix):
    '''为字典的键加上前缀.'''
    return {prefix + k: v for k, v in d.items()}

def format_dict(d, delimeter=', ', decimals=2):
    '''将{名称: 数值}结构的字典格式化为一行字符串.'''
    parts = [f'{k}={v:.{decimals}f}' for k, v in d.items()]
    info = delimeter.join(parts)

    return info

def to_iterable(x):
    '''若x不可迭代, 将其放入列表构成可迭代对象. 否则返回原对象.'''
    if isinstance(x, str) or not isinstance(x, Iterable):
        return [x]
    else:
        return x

def is_series(x):
    return isinstance(x, pd.Series)

def is_frame(x):
    return isinstance(x, pd.DataFrame)

def is_series_or_frame(x):
    return is_series(x) or is_frame(x)

def is_timeindex(x):
    return isinstance(x, pd.DatetimeIndex)

def is_timeseries(x):
    return is_series(x) and isinstance(x.index, pd.DatetimeIndex)

def is_timeframe(x):
    return is_frame(x) and isinstance(x.index, pd.DatetimeIndex)

def is_time_series_or_frame(x):
    return is_timeseries(x) or is_timeframe(x)

def check_series(*args):
    '''检查输入是否为Series.'''
    for arg in args:
        if not is_series(arg):
            raise ValueError('输入应该是Series')

def check_frame(*args):
    '''检查输入是否为DataFrame.'''
    for arg in args:
        if not is_frame(arg):
            raise ValueError('输入应该是DataFrame')

def check_series_or_frame(*args):
    '''检查输入是否为Series或DataFrame.'''
    for arg in args:
        if not is_series_or_frame(arg):
            raise ValueError('输入应该是Series或DataFrame')

def check_timeindex(*args):
    '''检查输入是否为DatetimeIndex.'''
    for arg in args:
        if not is_timeindex(arg):
            raise ValueError('输入应该是时间索引.')

def check_timeseries(*args):
    '''检查输入是否为带时间索引的Series.'''
    for arg in args:
        if not is_timeseries(arg):
            raise ValueError('输入应该是带时间索引的Series')

def check_timeframe(*args):
    '''检查输入是否为带时间索引的DataFrame.'''
    for arg in args:
        if not is_timeframe(arg):
            raise ValueError('输入应该是带时间索引的DataFrame')

def check_time_series_or_frame(*args):
    '''检查输入是否为带时间索引的Series或DataFrame.'''
    for arg in args:
        if not is_time_series_or_frame(arg):
            raise ValueError('输入应该是带时间索引的Series或DataFrame')

def pandas_like(x, y):
    '''若y是Series或DataFrame, 将x转换成与y相匹配的类型.'''
    if isinstance(y, pd.Series):
        return pd.Series(x, index=y.index)
    elif isinstance(y, pd.DataFrame):
        return pd.DataFrame(x, index=y.index, columns=y.columns)
    else:
        return x