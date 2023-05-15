from pathlib import Path
from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd
from pytz import timezone
from skyfield import api, almanac

from powerutils.common import (
    check_series,
    check_frame,
    check_series_or_frame,
    check_timeindex,
    check_timeseries
)

def try_read_csv(filepath, encodings=None, **kwargs):
    '''尝试在pd.read_csv中使用不同的编码. 默认使用UTF-8和GBK.'''
    if encodings is None:
        encodings = ['utf-8', 'gbk']

    for encoding in encodings:
        try:
            return pd.read_csv(str(filepath), encoding=encoding, **kwargs)
        except UnicodeDecodeError as e:
            exc = e
    raise exc

def read_table(filepath, **kwargs):
    '''用pd.read_excel读取Excel文件, 用try_read_csv读取其它文件.'''
    filepath = Path(filepath)
    if filepath.suffix in ['.xls', '.xlsx', '.et']:
        reader = pd.read_excel
    else:
        reader = try_read_csv
    df = reader(str(filepath), **kwargs)

    return df

def print_time_index(index):
    '''打印Series或DataFrame的时间索引的顺序和范围.'''
    check_timeindex(index)
    delta = index.to_series().diff().value_counts()

    print('[索引顺序]')
    print('是否重复:', index.has_duplicates)
    print('是否单增:', index.is_monotonic_increasing)
    print('是否等距:', len(delta) == 1)

    print('[时间范围]')
    print(index[0])
    print(index[-1])
    print(index[-1] - index[0])
    print('[时间间隔]')
    print(delta.sort_index())

def describe(df, decimals=2):
    '''添加了有效数据比例的DataFrame.describe.'''
    check_frame(df)
    ratio = (~df.isna()).sum(axis=0) / df.shape[0]
    table = df.describe().T
    table.insert(1, 'valid', ratio)
    table = table.round(decimals)

    return table

def sort_and_drop_duplicates(x, keep='last'):
    '''按索引排序并去重. 为了保证稳定性, 先去重再排序.'''
    check_series_or_frame(x)
    x = x.loc[lambda x: ~x.index.duplicated(keep)].sort_index()

    return x

def remove_column_units(df):
    '''去掉列名中的单位. 单位通过左括号分隔开来.'''
    check_frame(df)
    columns = df.columns.str.split('(', regex=False).str[0]
    columns = columns.str.split('（', regex=False).str[0]
    df = df.set_axis(columns, axis=1)

    return df

def calc_corr(df, key):
    '''计算某个变量和其它变量的相关性.'''
    check_frame(df)
    target = df[key]
    df = df.drop(columns=key)
    corr = df.corrwith(target).sort_values().rename(key)

    return corr

def combination_corr(df, key):
    '''计算某个变量和其它列两两组合后的相关性. 组合包括加减乘除.'''
    check_frame(df)
    target = df[key]
    df = df.drop(columns=key)
    data = {}

    # 使pandas除法中产生的inf变为NaN.
    with pd.option_context('mode.use_inf_as_na', True):
        for name1, name2 in combinations(df.columns, 2):
            s1 = df[name1]
            s2 = df[name2]
            c1 = s1.corr(target)
            c2 = s2.corr(target)

            c_add = (s1 + s2).corr(target)
            data[f'{name1}+{name2}'] = (c1, c2, c_add)

            c_sub = (s1 - s2).corr(target)
            data[f'{name1}-{name2}'] = (c1, c2, c_sub)

            c_mul = (s1 * s2).corr(target)
            data[f'{name1}*{name2}'] = (c1, c2, c_mul)

            c_div1 = (s1 / s2).corr(target)
            data[f'{name1}/{name2}'] = (c1, c2, c_div1)

            c_div2 = (s2 / s1).corr(target)
            data[f'{name2}/{name1}'] = (c2, c1, c_div2)

    df_corr = pd.DataFrame.from_dict(
        data, orient='index', columns=['左', '右', '组合']
    ).sort_values('组合')

    return df_corr

def mask_data_by_n_sigma(x, n_sigma=3, other=np.nan):
    '''将n个标准差范围外的数据设为缺测.'''
    check_series_or_frame(x)
    mean = x.mean()
    std = x.std()
    lower = mean - n_sigma * std
    upper = mean + n_sigma * std
    cond = (x >= lower) & (x <= upper)

    return x.where(cond, other)

def _get_interval_cond(interval):
    '''返回判断是否落入区间的匿名函数.'''
    minval, maxval = interval
    minval = -np.inf if minval is None else minval
    maxval = np.inf if maxval is None else maxval
    cond = lambda x: (x >= minval) & (x <= maxval)

    return cond

def mask_data_outside_interval(df, interval, other=np.nan):
    '''
    给出形如(minval, maxval)的区间, 将区间外的数据设为缺测.

    minval为None时表示无穷小.
    maxval为None时表示无穷大.
    '''
    check_frame(df)
    df = df.copy()
    if isinstance(interval, dict):
        for key, itv in interval.items():
            cond = _get_interval_cond(itv)
            df[key].where(cond, other, inplace=True)
    else:
        cond = _get_interval_cond(interval)
        df.where(cond, other, inplace=True)

    return df

def count_consecutive_values(series, cond):
    '''
    统计序列中某类元素连续出现的次数.

    特定元素通过cond或cond(series)给出的布尔数组选出.
    结果序列的长度和series相同, 非特定元素对应的计数为0,
    特定元素对应的计数为连续出现的次数.
    '''
    check_series(series)
    not_value_mask = ~cond(series) if callable(cond) else ~cond
    value_id = (not_value_mask.diff() > 0).cumsum() + 1
    value_id[not_value_mask] = 0
    value_count = value_id.groupby(value_id).transform('count')
    value_count[not_value_mask] = 0

    return value_count

def count_consecutive_zeros(series):
    '''统计连续出现零值的次数.'''
    return count_consecutive_values(series, cond=lambda x: x <= 0)

def count_consecutive_nan(series):
    '''统计连续出现NaN的次数.'''
    return count_consecutive_values(series, cond=lambda x: x.isna())

def interp_gaps(series, method='linear', max_gap_len=1):
    '''插值填补长度不超过max_gap_len的缺口. 缺口由NaN组成.'''
    check_series(series)
    series = (series
        .interpolate(method=method, limit_area='inside')
        .where(count_consecutive_nan(series) <= max_gap_len)
    )

    return series

def wd_to_cos_sin(wd):
    '''计算风向的cos和sin值.'''
    wd = np.deg2rad(270 - wd)
    cos = np.cos(wd)
    sin = np.sin(wd)

    return cos, sin

def wswd_to_uv(ws, wd):
    '''风速风向转为uv分量.'''
    cos, sin = wd_to_cos_sin(wd)
    u = ws * cos
    v = ws * sin

    return u, v

def uv_to_wswd(u, v):
    '''uv分量转为风速风向.'''
    ws = np.hypot(u, v)
    wd = np.rad2deg(np.arctan2(u, v)) + 180

    return ws, wd

def make_time_cos_sin(
    index, day=True, week=False, twoweek=False,
    month=False, halfyear=False, year=False
):
    '''
    计算时间周期的余弦和正弦特征构成的DataFrame.

    day表示正弦和余弦特征的周期为一天.
    week表示一周, 其它参数依此类推.
    '''
    check_timeindex(index)
    len_day = 24 * 60 * 60
    len_week = 7 * len_day
    len_twoweek = 2 * len_week
    len_year = 365.2425 * len_day
    len_month = len_year / 12
    len_halfyear = len_year / 2
    seconds = index.map(pd.Timestamp.timestamp)
    x = seconds * 2 * np.pi

    df = pd.DataFrame()
    if day:
        df['day_cos'] = np.cos(x / len_day)
        df['day_sin'] = np.sin(x / len_day)
    if week:
        df['week_cos'] = np.cos(x / len_week)
        df['week_sin'] = np.sin(x / len_week)
    if twoweek:
        df['twoweek_cos'] = np.cos(x / len_twoweek)
        df['twoweek_sin'] = np.sin(x / len_twoweek)
    if month:
        df['month_cos'] = np.cos(x / len_month)
        df['month_sin'] = np.sin(x / len_month)
    if halfyear:
        df['halfyear_cos'] = np.cos(x / len_halfyear)
        df['halfyear_sin'] = np.sin(x / len_halfyear)
    if year:
        df['year_cos'] = np.cos(x / len_year)
        df['year_sin'] = np.sin(x / len_year)
    # 结果是默认索引, 需要修改.
    df.set_index(index, inplace=True)

    return df

def make_hour_encoding(index):
    '''生成24个小时的独热编码的DataFrame.'''
    check_timeindex(index)
    df = (pd.get_dummies(index.hour, prefix='hour')
        .reindex(columns=[f'hour_{i}' for i in range(24)])
        .set_index(index)
        .fillna(0)
    )

    return df

filepath = Path(__file__).parent / 'de421.bsp'

def is_daytime(index, lon, lat):
    '''根据输入的时间索引计算是否为白天的mask.'''
    check_timeindex(index)
    ts = api.load.timescale()
    eph = api.load(str(filepath))
    obs = api.wgs84.latlon(lat, lon)
    func = almanac.sunrise_sunset(eph, obs)

    tz = timezone('Asia/Shanghai')
    time = ts.from_datetimes(index.tz_localize(tz))
    mask = pd.Series(func(time), index)

    return mask

def sun_altitude(index, lon, lat):
    '''根据输入的时间索引计算太阳高度角. 单位为弧度.'''
    check_timeindex(index)
    ts = api.load.timescale()
    eph = api.load(str(filepath))
    sun = eph['sun']
    earth = eph['earth']
    obs = earth + api.wgs84.latlon(lat, lon)

    tz = timezone('Asia/Shanghai')
    time = ts.from_datetimes(index.tz_localize(tz))
    alt = obs.at(time).observe(sun).apparent().altaz()[0].radians
    alt = pd.Series(alt, index=index)

    return alt

def _to_lags(lags):
    '''将lags转为可迭代对象.'''
    if isinstance(lags, Iterable):
        return lags
    else:
        if lags < 0:
            return range(-lags, 1)
        else:
            return range(0, lags + 1)

def make_lags(series, lags=-1, freq=None, align=False, name=None):
    '''
    创建滞后序列构成的DataFrame.

    结果的列名形如{name}(t=-1)和{name}(t=1).
    t<0表示过去时刻, t=0表示无滞后, t>0表示未来时刻.
    lags为负数时, 构造[lags, 0]的滞后.
    lags为正数时, 构造[0, lags]的滞后.
    lags为可迭代对象时, 采用元素的值构造滞后.
    align指定结果的索引与series对齐.
    '''
    check_timeseries(series)
    if name is None:
        name = getattr(series, 'name', '')
    lags = _to_lags(lags)

    data = {}
    for lag in lags:
        key = f'{name}(t={lag})'
        data[key] = series.shift(-lag, freq)

    df = pd.DataFrame(data)
    if align:
        df = df.reindex(series.index)

    return df

def make_diffs(series, lags=-1, freq=None, align=False, name=None):
    '''
    创建差分序列构成的DataFrame.

    结果的列名形如{name}(dt=-1)和{name}(dt=1).
    dt<0表示后向差分, dt=0表示自己减自己, dt>0表示正向差分.
    其它参数类似make_lags函数.
    '''
    check_timeseries(series)
    if name is None:
        name = getattr(series, 'name', '')
    lags = _to_lags(lags)

    data = {}
    for lag in lags:
        key = f'{name}(dt={lag})'
        lagged = series.shift(-lag, freq)
        if lag < 0:
            data[key] = series - lagged
        else:
            data[key] = lagged - series

    df = pd.DataFrame(data)
    if align:
        df = df.reindex(series.index)

    return df

def make_rollings(series, window, center=False, name=None, win_info=False):
    '''
    创建滚动统计量构成的DataFrame.

    结果的列名形如{name}(min)和{name}(max).
    win_info为True时会使列名形如{name}({window}.min).
    '''
    check_timeseries(series)
    if name is None:
        name = getattr(series, 'name', '')

    if win_info:
        labeler = lambda x: f'{name}({window}.{x})'
    else:
        labeler = lambda x: f'{name}({x})'

    df = pd.DataFrame()
    roller = series.rolling(window, center=center)
    df[labeler('min')] = roller.min()
    df[labeler('max')] = roller.max()
    df[labeler('sum')] = roller.sum()
    df[labeler('avg')] = roller.mean()
    df[labeler('std')] = roller.std()
    df[labeler('var')] = roller.var()

    return df