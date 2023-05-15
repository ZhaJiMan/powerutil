import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

from powerutils.common import (
    check_timeseries,
    check_timeframe,
    check_time_series_or_frame,
    to_iterable
)

def plot_missing(df, freq='M', figsize=(8, 6), labelsize='small'):
    '''画出DataFrame数据缺失的情况. freq指定刻度间隔.'''
    check_timeframe(df)
    df = df.asfreq('15T')
    ax = msno.matrix(df, freq=freq, sparkline=False, figsize=figsize)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, ha='center', rotation=90)
    ax.tick_params(labelsize=labelsize)

    return ax

def plot_timeseries(
    x, start_time=None, end_time=None,
    figsize=None, labelsize='small'
):
    '''在一张子图上画出时间序列.'''
    check_time_series_or_frame(x)
    if figsize is None:
        figsize = (10, 4)
    _, ax = plt.subplots(figsize=figsize)
    x = x.loc[start_time:end_time].asfreq('15T')
    x.plot(ax=ax, lw=1, alpha=0.8, xlabel='')
    ax.tick_params(labelsize=labelsize)

    return ax

def plot_multi_timeseries(
    df, keys, filter=False,
    start_time=None, end_time=None,
    figsize=None, sharex=False, labelsize='small'
):
    '''
    在多张子图上画出多条时间序列.

    filter=True时, 会通过df.filter(like=key)选出要画的序列. 要求keys形如:
    [key1, key2, key3]
    filter=False时, 会通过df[key]选出要画的序列. 要求keys形如:
    [[key11, key12], [key21, key22], [key31, key32]]
    '''
    check_timeframe(df)
    nrows = len(keys)
    if figsize is None:
        figsize = (10, 3 * nrows)
    _, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=sharex)
    axes = to_iterable(axes)

    df = df.loc[start_time:end_time].asfreq('15T')
    for key, ax in zip(keys, axes):
        if filter:
            dfa = df.filter(like=key)
        else:
            dfa = df[to_iterable(key)]
        dfa.plot(ax=ax, lw=1, alpha=0.8)
        ax.legend(loc='upper right', fontsize=labelsize)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
        ax.set_xlabel('')
        ax.tick_params(labelsize=labelsize)

    return axes

def plot_twin_timeseries(
    df, left_key, right_key,
    start_time=None, end_time=None,
    left_color=None, right_color=None,
    figsize=(10, 4), labelsize='small'
):
    '''
    在一张图上画出两种y轴的时间序列图.

    left_key可以是列名或列名组成的列表.
    right_key可以是列名或列名组成的列表.
    '''
    check_timeframe(df)
    left_keys = to_iterable(left_key)
    right_keys = to_iterable(right_key)
    if left_color is None:
        left_colors = ['navy', 'dodgerblue']
    else:
        left_colors = to_iterable(left_color)
    if right_color is None:
        right_colors = ['darkred', 'crimson']
    else:
        right_colors = to_iterable(right_color)

    df = df.loc[start_time:end_time].asfreq('15T')
    _, ax1 = plt.subplots(figsize=figsize)
    df[left_keys].plot.line(ax=ax1, color=left_colors, legend=False)
    ax2 = ax1.twinx()
    df[right_keys].plot.line(ax=ax2, color=right_colors, legend=False)
    # ax1.legend会被ax2遮挡.
    ax2.legend(
        handles=[*ax1.lines, *ax2.lines],
        loc='upper right',
        fontsize=labelsize
    )

    for ax in [ax1, ax2]:
        ax.set_xlabel('')
        ax.tick_params(labelsize=labelsize)

    return ax1, ax2

def plot_daily_acc(
    acc, level=80, start_time=None, end_time=None,
    figsize=(10, 4), markersize=4, labelsize='small'
):
    '''画出日准确率围绕level的散点图.'''
    check_timeseries(acc)
    acc = acc.loc[start_time:end_time].asfreq('D')
    _, ax = plt.subplots(figsize=figsize)
    acc.plot(ax=ax, c='dimgray')
    acc.loc[acc >= level].plot(
        ax=ax, ls='none',
        marker='o', ms=markersize, mfc='limegreen', mec='k',
        label='合格'
    )
    acc.loc[acc < level].plot(
        ax=ax, ls='none',
        marker='o', ms=markersize, mfc='orangered', mec='k',
        label='不合格'
    )
    ax.legend(loc='lower left', fontsize=labelsize)
    ax.axhline(level, c='C3', ls='--', lw=1)
    ax.set_xlabel('')
    ax.set_ylabel('准确率(%)')
    ax.set_ylim(None, 100)
    ax.tick_params(labelsize=labelsize)

    return ax