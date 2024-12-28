import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from scipy.stats import norm, johnsonsu
import pywt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr


def compute_pooled_metrics(df: pd.DataFrame, levels: list):
    """Computes 4 metrics pooled by a list of levels."""
    outputs = {}
    for level in levels:
        metric_pool_df = pd.DataFrame(
            index=df.index.get_level_values(level).unique()
        )
        args = {'by': level, 'group_keys': False}
        metric_pool_df['rmse'] = df.groupby(**args).apply(
            lambda x: root_mean_squared_error(*x.values.T)
        )
        metric_pool_df['mae'] = df.groupby(**args).apply(
            lambda x: mean_absolute_error(*x.values.T)
        )
        metric_pool_df['pearson_r'] = df.groupby(**args).apply(
            lambda x: pearsonr(*x.values.T)[0]
        )
        metric_pool_df['spearman_r'] = df.groupby(**args).apply(
            lambda x: spearmanr(*x.values.T)[0]
        )
        outputs[f'{level}_pool'] = metric_pool_df
    return outputs


def pandas_winsor(series, limit=1 - 0.99):
    return pd.Series(winsorize(series, limits=limit).data, index=series.index)


def Winsorize_robustZ(series: pd.Series, fraction):
    """Winsorizes the input series based on robust Z-scores."""
    mad = np.abs(series - series.median()).median()
    robust_zscore = np.abs(series - series.median()) / mad
    zscore_quantile = max(3, min(robust_zscore.quantile(fraction), 10))
    winsor_limit = (robust_zscore >= zscore_quantile).mean()
    winsorized_zscore = winsorize(
        robust_zscore, limits=winsor_limit, nan_policy='omit'
    )
    winsorized_series = pd.Series(
        winsorized_zscore * mad * np.sign(series - series.median()).values,
        index=series.index,
    )
    winsorized_series += series.median()
    output = {
        'winsorized_series': winsorized_series,
        'zscore_quantile': zscore_quantile,
        'winsor_limit': winsor_limit,
    }
    return pd.DataFrame(output, index=series.index)


def WinsorizeReturns_robustZ(series: pd.Series, fraction):
    """An implementation of robust winsorization specific for returns."""
    log_return = series.apply(np.log1p)
    mad = log_return.apply(np.abs).median()
    robust_zscore = np.abs(log_return) / mad
    zscore_quantile = max(5, min(robust_zscore.quantile(fraction), 10))
    winsor_limit = (robust_zscore >= zscore_quantile).mean()
    winsorized_zscore = winsorize(
        robust_zscore, limits=winsor_limit, nan_policy='omit'
    )
    winsorized_series = pd.Series(
        winsorized_zscore * mad * np.sign(log_return).values,
        index=series.index,
    )
    output = {
        'winsorized_series': winsorized_series,
        'zscore_quantile': zscore_quantile,
        'winsor_limit': winsor_limit,
    }
    return pd.DataFrame(output, index=series.index)


def wavelet_smooth(signal, threshold=0.2, level=3, wavelet='db6'):
    coeff = pywt.wavedec(signal, wavelet, mode='per', level=level)
    coeff[1:] = [
        pywt.threshold(
            i, value=threshold * np.abs(signal.max()), mode='soft'
        )
        for i in coeff[1:]
    ]
    clean_signal = pywt.waverec(coeff, wavelet, mode='per')
    if len(signal) % 2 == 1:
        clean_series = pd.Series(clean_signal[:-1], signal.index)
    else:
        clean_series = pd.Series(clean_signal, signal.index)
    return clean_series


def rolling_beta(df: pd.DataFrame, rolling_window, column, market_df):
    Y_series = df[column]
    X_series = market_df[column]
    XY_series = X_series * Y_series
    XY_mean = XY_series.rolling(window=rolling_window).mean()
    X_mean = X_series.rolling(window=rolling_window).mean()
    Y_mean = Y_series.rolling(window=rolling_window).mean()
    X_variance = X_series.rolling(window=rolling_window).var()
    return (XY_mean - X_mean * Y_mean) / X_variance


def johnson_su_transform(series: pd.Series):
    jsu_params = johnsonsu.fit(series.dropna())
    uniform_values = johnsonsu.cdf(series.dropna(), *jsu_params)
    normal_transformed_values = norm.ppf(uniform_values)
    output = pd.Series(normal_transformed_values, index=series.dropna().index)
    return output, jsu_params


def johnson_su_inverse_transform(transformed_series: pd.Series, jsu_params):
    uniform_values = norm.cdf(transformed_series)
    original_values = johnsonsu.ppf(uniform_values, *jsu_params)
    return pd.Series(original_values, index=transformed_series.index)


def tail_specific_transform(series, left_threshold, transformation='log'):
    transformed_series = series.copy()
    if transformation == 'log':
        left_tail = transformed_series < left_threshold
        transformed_series[left_tail] = np.log1p(
            np.abs(transformed_series[left_tail])
        ) * np.sign(transformed_series[left_tail])
    return transformed_series


def ewm_rolling_beta(
        df: pd.DataFrame, column, ewm_alpha, min_periods, market_df):
    Y_series = df[column]
    X_series = market_df[column]
    XY_series = X_series * Y_series
    ewm_params = {'alpha': ewm_alpha, 'min_periods': min_periods}
    XY_mean = XY_series.ewm(**ewm_params).mean()
    X_mean = X_series.ewm(**ewm_params).mean()
    Y_mean = Y_series.ewm(**ewm_params).mean()
    X_variance = X_series.ewm(alpha=ewm_alpha).var()
    return (XY_mean - X_mean * Y_mean) / X_variance


class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs."""

    def __init__(
        self,
        n_splits=3,
        train_period_length=121,
        test_period_length=21,
        lookahead=None,
        shuffle=False,
    ):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(1).unique()
        days = sorted(unique_dates, reverse=True)

        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = (
                train_end_idx + self.train_length + self.lookahead - 1
            )
            split_idx.append(
                [train_start_idx, train_end_idx, test_start_idx, test_end_idx]
            )

        dates = X.reset_index()['Date']
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[
                (dates > days[train_start]) & (dates <= days[train_end])
            ].index
            test_idx = dates[
                (dates > days[test_start]) & (dates <= days[test_end])
            ].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits
