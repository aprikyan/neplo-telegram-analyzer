import datetime
import pandas as pd
import streamlit


def contains_link(message):
    if type(message) == list:
        for i in message:
            if type(i) == dict and i["type"] == "link":
                return True
    return False


def dayify(date, day_start=6):
    return date.date() - datetime.timedelta(days=1) if date.hour <= day_start else date.date()


def hide_zeros(series):
    return series[series != 0]


def complement_index(series, fill_value=None):
    d1, d2 = series.index.min(), series.index.max()
    missed_indices = [d1 + i * datetime.timedelta(days=1) for i in range((d2 - d1).days)]
    return series.append(
        pd.Series({date: fill_value for date in missed_indices if date not in series.index})).sort_index()


def groupby(series, period="month", aggfunc="sum", normalize=False, fill_value=0):
    multiindex = type(series.index) == pd.core.indexes.multi.MultiIndex
    if multiindex:
        df = series.unstack().fillna(fill_value)
    else:
        df = pd.DataFrame(series)
        df.columns = ["data"]
    cols = df.columns

    df["year"] = pd.Series(df.index).apply(getattr, args=["year"]).values
    period_cols = ["year"]
    if period == "month":
        df["month"] = pd.Series(df.index).apply(getattr, args=["month"]).values
        period_cols.append("month")
    elif period == "week":
        df["week"] = pd.Series(df.index).apply(lambda x: datetime.date.isocalendar(x)[1]).values
        period_cols.append("week")

    func = {"sum": sum, "len": len, "mean": pd.Series.mean, "max": max, "counts": pd.Series.value_counts}[aggfunc]
    df = df.groupby(period_cols)[cols].apply(func)
    if normalize:
        df = df.fillna(fill_value).unstack().apply(lambda x: x / x.sum(), axis=1).stack()
    return df


def smoothen(series, window=5):
    #     return pd.Series([series.iloc[i:i+10].mean() for i in range(len(series) - window)])
    return series.rolling(window).mean().dropna()


def a_vs_b(a_s, b_s, func, agg_func=sum, **kwargs):
    outs = []
    for params_set in (a_s, b_s):
        if len(params_set) == 1:
            agg_func = lambda x: x[0]
        if type(params_set[0]) == dict:
            out = agg_func([func(**params, **kwargs) for params in params_set])
        else:
            out = agg_func([func(params, **kwargs) for params in params_set])
        outs.append(out)
    return outs


def mod(series, return_counts=False):
    series = pd.Series(series)
    series = series.value_counts()
    if len(series):
        return (series.index[0], series.iloc[0]) if return_counts else series.index[0]
    return None


def flatten_multiindex(series, aggfunc=sum, fill_value=0):
    return series.unstack().fillna(fill_value).apply(aggfunc, axis=1)


def consecutive_days(series):
    series = pd.Series(series).sort_values()
    day = datetime.timedelta(days=1)
    series.index = range(len(series))
    interruptions = series.index[series.diff() > day].tolist()
    return [list(range(i, j)) for i, j in zip([0] + interruptions, interruptions + [len(series)])]


def longest_consecutive_days(series):
    series = pd.Series(series).sort_values()
    longest = max(consecutive_days(series), key=len)
    return (series[longest[0]], series[longest[-1]])