import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from scipy.stats.mstats import winsorize
from scipy.stats import probplot
from scipy.stats import norm, johnsonsu
import pywt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr



idx = pd.IndexSlice

### PLOTTING
def plot_qq(series, dist='norm', title='Q-Q Plot', dist_params=None):
    """
    Generates a Q-Q plot for the given pandas Series, transformed according to the distribution.
    
    Parameters:
    - series (pd.Series): The data series to generate the Q-Q plot for. May contain NaNs.
    - dist (str): The theoretical distribution to compare against. Default is 'norm' (normal distribution).
    - title (str): Title of the Q-Q plot.
    - dist_params (tuple): Optional parameters for the distribution (for custom distributions like 'johnsonsu').
    """
    # Drop NaN values and ensure input is a pandas Series
    if not isinstance(series, pd.Series): 
        series = pd.Series(series)
    clean_series = series.dropna()
    
    # If the target distribution is 'norm', standardize the series
    if dist == 'norm':
        standardized_series = (clean_series - clean_series.mean()) / clean_series.std()
    
    # If using a custom distribution (e.g., 'johnsonsu', 't'), apply the CDF and PPF to transform the data
    else:
        if dist_params is None:
            raise ValueError(f"Distribution parameters (dist_params) must be provided for {dist}.")
        
        # Apply the CDF of the data (based on the fitted distribution)
        cdf_values = stats.__dict__[dist].cdf(clean_series, *dist_params)
        # Apply the PPF of the theoretical distribution to get the transformed quantiles
        standardized_series = stats.norm.ppf(cdf_values)

    # Generate Q-Q plot using the transformed series
    fig = plt.figure()
    
    # For custom distributions, we apply the correct distribution-specific transformations
    (theoretical_quantiles, ordered_values), regression_line = probplot(standardized_series, dist=dist, sparams=dist_params)
    
    # Plot the Q-Q points
    plt.plot(theoretical_quantiles, ordered_values, 'o', label='Data Points')
    
    # Plot the line of best fit
    plt.plot(theoretical_quantiles, regression_line[0] * theoretical_quantiles + regression_line[1], 'r-', label='Best Fit Line')
    
    # Add the y = x reference line
    plt.plot(theoretical_quantiles, theoretical_quantiles, 'k--', label='y = x (Reference Line)')
    
    # Add labels, title, and legend
    plt.title(title)
    plt.xlabel(f'Theoretical Quantiles of {dist} distribution')
    plt.ylabel(f'Ordered Values from {clean_series.name}')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()


def plot_actual_vs_predicted(train_df, holdout_df, model_name='Model', actual_col='actual_target', predicted_col='target_prediction'):
    """
    Plots actual vs. predicted values for training and holdout sets,
    and computes Pearson r, RMSE, and MAE for both sets.

    Parameters:
    - train_df: DataFrame containing the training set with actual and predicted values.
    - holdout_df: DataFrame containing the holdout set with actual and predicted values.
    - model_name: String representing the name of the model (for plot titles).
    - actual_col: Name of the column with actual target values.
    - predicted_col: Name of the column with predicted values.
    """
    # Compute metrics for training set
    y_train_actual = train_df[actual_col]
    y_train_pred = train_df[predicted_col]
    pearson_train = pearsonr(y_train_actual, y_train_pred)[0]
    rmse_train = mean_squared_error(y_train_actual, y_train_pred, squared=False)
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)

    # Compute metrics for holdout set
    y_holdout_actual = holdout_df[actual_col]
    y_holdout_pred = holdout_df[predicted_col]
    pearson_holdout = pearsonr(y_holdout_actual, y_holdout_pred)[0]
    rmse_holdout = mean_squared_error(y_holdout_actual, y_holdout_pred, squared=False)
    mae_holdout = mean_absolute_error(y_holdout_actual, y_holdout_pred)

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    axa, axb = ax

    # Plot for training set
    sns.scatterplot(x=y_train_pred, y=y_train_actual, alpha=0.3, ax=axa)
    axa.set_title(f'TRAINING SET\n{model_name}\nPearson r={pearson_train:.3g}\n'+f'RMSE: {rmse_train:.3f}\nMAE: {mae_train:.3f}', fontsize=14)
    axa.set_xlabel('Predicted Values')
    axa.set_ylabel('Actual Values')

    

    # Add metrics text
    # axa.text(0.95, 0.95, f'RMSE: {rmse_train:.3f}\nMAE: {mae_train:.3f}',
            #  transform=axa.transAxes, fontsize=12, verticalalignment='top')

    # Plot for holdout set
    sns.scatterplot(x=y_holdout_pred, y=y_holdout_actual, alpha=0.3, ax=axb)
    axb.set_title(f'HOLDOUT SET\n{model_name}\nPearson r={pearson_holdout:.3g}\n'+f'RMSE: {rmse_holdout:.3f}\nMAE: {mae_holdout:.3f}', fontsize=14)
    axb.set_xlabel('Predicted Values')
    axb.set_ylabel('Actual Values')

    # Add metrics text
    # axb.text(0.95, 0.95, f'RMSE: {rmse_holdout:.3f}\nMAE: {mae_holdout:.3f}',
    #          transform=axb.transAxes, fontsize=12, verticalalignment='top')

    # Plot y = x line
    for a in [axa, axb]:
        a.set_xlim(a.get_xlim())
        a.set_ylim(a.get_ylim())
        a.plot(a.get_xlim(), a.get_xlim(), '--', color='tab:orange', linewidth=3, label='y=x')
    
    axa.legend(fontsize=12)



    # Display metrics
    print("Training Set Metrics:")
    print(f"Pearson r: {pearson_train:.3f}")
    print(f"RMSE: {rmse_train:.3f}")
    print(f"MAE: {mae_train:.3f}\n")

    print("Holdout Set Metrics:")
    print(f"Pearson r: {pearson_holdout:.3f}")
    print(f"RMSE: {rmse_holdout:.3f}")
    print(f"MAE: {mae_holdout:.3f}")


def compute_pooled_metrics(df:pd.DataFrame, levels:list):
    '''
    Computes 4 metrics pooled by a list of levels in a pandas dataframe.
    df should only have two columnes, the predcited and the actual target values.
    '''
    outputs = {}
    for level in levels:
        metric_pool_df = pd.DataFrame(
            index = df.index.get_level_values(level).unique())

        metric_pool_df['rmse'] = df.groupby(level, group_keys=False).apply(
            lambda x: mean_squared_error(*x.values.T, squared=False))         
        metric_pool_df['mae'] = df.groupby(level, group_keys=False).apply(
            lambda x: mean_absolute_error(*x.values.T)) 
        metric_pool_df['pearson_r'] = df.groupby(level, group_keys=False).apply(
            lambda x: pearsonr(*x.values.T)[0])
        metric_pool_df['spearman_r'] = df.groupby(level, group_keys=False).apply(
            lambda x: spearmanr(*x.values.T)[0])  
    
        outputs[f'{level}_pool'] = metric_pool_df
    
    return outputs

def plot_time_metrics(baseline_metrics_df, metrics_df:pd.DataFrame, model_name:str):
    assert (baseline_metrics_df.columns == metrics_df.columns).all()
    columns = metrics_df.columns
    n = len(columns)
    fig, ax = plt.subplots(n, 1, figsize=(9.2, 5.6*n*2/3))
    fig.subplots_adjust(hspace= 0.5)
    for i, col in enumerate(columns):
        metric_mean = metrics_df[col].mean()
        baseline_metric_mean = baseline_metrics_df[col].mean()
        ax[i].set_title('Mean daily baseline {:} = {:.3g}\nMean daily {:} {:} = {:.3g}'.format(
            col, baseline_metric_mean, model_name, col, metric_mean))
        ax[i].axhline(baseline_metric_mean, color='tab:blue', linestyle='--')
        ax[i].axhline(metric_mean, color='tab:orange', linestyle='--')
        
        sns.lineplot(baseline_metrics_df, x=metrics_df.index.name, y=col, ax=ax[i], marker='o', label='baseline')
        sns.lineplot(metrics_df, x=metrics_df.index.name, y=col, ax=ax[i], marker='o', label=model_name)
        ax[i].legend()
    return
    
# Feature calculation functions

def pandas_winsor(series, limit = 1-0.99):
    return pd.Series(winsorize(series, limits = limit).data, index =series.index)


# Plotting methods

def feature_timeseries_plot(tickers:list, feature, target, df:pd.DataFrame = None):
    if df == None: df = train

    fig, ax = plt.subplots(2, len(tickers), figsize = (14, 10))

    for i , tick in enumerate(tickers):
        ax[0, i].set_title('Security code {:}'.format(tick))
        ax[1, i].set_title('beta rank {:}'.format(universe_stats.loc[tick, 'beta_rank']))        
        sns.lineplot(df.loc[tick], x= 'Date', y = target, ax = ax[0, i])
        sns.lineplot(df.loc[tick], x= 'Date', y = feature, ax = ax[1, i])
    return

# def beta_feature_scatter(feature, spearman = False, df:pd.DataFrame = None,):
#     '''
#     2 scatter plots of the beta and beta differences vs. the feature
#     '''
#     fig, ax = plt.subplots(ncols= 2, figsize = (9.2*1.5,5.6))
#     feature_scatter(feature, 'beta_target_45', spearman, df, fig =fig, ax = ax[0])

def feature_scatter(feature, target, spearman=False, df:pd.DataFrame=None, fig =None, ax = None):

    if df ==None:
        df = train.loc[train.index.get_level_values(0)!=1320]
    if ax ==None:
        fig, ax = plt.subplots()
    df=df[[feature, target]].dropna()
    sns.scatterplot(data = df, x = feature, y = target, ax = ax)

    pearson_r = pearsonr(df[feature], df[target])[0]
    print(f'{target} vs {feature} pearson r={pearson_r}')
    ax.set_title('{:} has pearson r of {:.3g}'.format(feature, pearson_r))

    if spearman:
        spearman_r = spearmanr(df[feature], df[target])[0]
        print(f'{target} vs {feature} spearman r={spearman_r}')
    
def feature_heatmap(feat_list:list,targets:list = None, data:pd.DataFrame =None, ):
    '''
    Plots a heat map of the correaltion matrix
    ''' 
    n_feats = len(feat_list)
    fig, ax = plt.subplots(figsize = (1.5 * n_feats, 1.5*n_feats))

    if data == None: data = train
    if targets == None: targets = ['beta_target_45']
    cols = feat_list + targets
    sns.heatmap(data[cols].dropna().corr(), ax = ax, cmap = 'viridis',annot=True, fmt = '.2g')
        


### PREPROCESSING 
def Winsorize_robustZ(series: pd.Series, fraction):
    """
    Winsorizes the input series based on robust Z-scores, computed from the median absolute deviation (MAD).
    
    Parameters:
    - series (pd.Series): Input data series to be winsorized.
    - fraction (float): The fraction of the data to be winsorized (e.g., 0.997).

    Returns:
    - pd.DataFrame: DataFrame containing the winsorized series, z-score quantile, and winsor limit.
    """
    # Calculate the median absolute deviation (MAD)
    mad = np.abs(series - series.median()).median()
    
    # Compute the robust Z-score
    robust_zscore = np.abs(series - series.median()) / mad
    
    # Determine the quantile for winsorization
    zscore_quantile = robust_zscore.quantile(fraction)
    
    # Ensure the quantile is between the defined bounds of 5 and 10
    zscore_quantile = max(3, min(zscore_quantile, 10))
    
    # Calculate the winsorization limit
    winsor_limit = (robust_zscore >= zscore_quantile).mean()
    
    # Winsorize the series based on the robust Z-score
    winsorized_zscore = winsorize(robust_zscore, limits=winsor_limit, nan_policy='omit')
    
    # Scale back to the original series
    winsorized_series = pd.Series(
        winsorized_zscore * mad * np.sign(series - series.median()).values + series.median(),
        index=series.index)
    
    # Return a DataFrame containing the winsorized series, quantile, and limit
    output = {
        'winsorized_series': winsorized_series,
        'zscore_quantile': zscore_quantile,
        'winsor_limit': winsor_limit
    }
    return pd.DataFrame(output, index=series.index)



def WinsorizeReturns_robustZ(series:pd.Series, fraction):
    log_return = series.apply(np.log1p)
    mad = log_return.apply(np.abs).median()

    # mind the absolute value sign on the numerator
    robust_zscore = np.abs(log_return) / mad

    # Use a single tailed quanitle
    zscore_quantile = robust_zscore.quantile(fraction) 

    # The quantile to clip should be no smaller than 5,
    # and no larger than 10 (ref: EQI Gappy)
    zscore_quantile = max(5, min(zscore_quantile, 10))
    winsor_limit = (robust_zscore >= zscore_quantile).mean()

    winsorized_zscore = winsorize(robust_zscore, limits=winsor_limit, nan_policy='omit')
    winsorized_series = pd.Series(
        winsorized_zscore * mad * np.sign(log_return).values,
        index=series.index)
    output = {
        'winsorized_series': winsorized_series,
        'zscore_quantile': zscore_quantile,
        'winsor_limit': winsor_limit
    }
    return pd.DataFrame(output, index=series.index)  

def wavelet_smooth(signal, threshold = 0.2, level =3, wavelet = 'db6'):

    coeff = pywt.wavedec(signal, wavelet, mode = 'per', level =level)
    coeff[1:]  = [pywt.threshold(i, value=threshold*np.abs(signal.max()), mode = 'soft') for i in coeff[1:]]
    clean_signal = pywt.waverec(coeff, wavelet, mode = 'per')
    # return clean_signal
    if len(signal)%2 == 1:
        clean_series= pd.Series(clean_signal[:-1], signal.index)
    else:
        clean_series =  pd.Series(clean_signal, signal.index)
    return clean_series

def rolling_beta(df:pd.DataFrame, rolling_window, column, market_df=market_df):
    Y_series= df[column]
    X_series = market_df[column]
    XY_series = X_series * Y_series
    XY_mean = XY_series.rolling(window=rolling_window).mean()
    X_mean = X_series.rolling(window=rolling_window).mean()
    Y_mean = Y_series.rolling(window=rolling_window).mean()
    X_variance = X_series.rolling(window=rolling_window).var()
    return (XY_mean - X_mean * Y_mean) / X_variance


def johnson_su_transform(series: pd.Series):
    """
    Transforms a pandas series to a more normal distribution using Johnson SU.
    
    Parameters:
    - series (pd.Series): Input data series to be transformed.
    
    Returns:
    - transformed_series (pd.Series): Transformed series that is more normally distributed.
    - jsu_params (tuple): Parameters of the fitted Johnson SU distribution.
    """
    jsu_params = johnsonsu.fit(series.dropna())
    uniform_values = johnsonsu.cdf(series.dropna(), *jsu_params)
    normal_transformed_values = norm.ppf(uniform_values)
    transformed_series = pd.Series(normal_transformed_values, index=series.dropna().index)

    return transformed_series, jsu_params

def johnson_su_inverse_transform(transformed_series: pd.Series, jsu_params: tuple):
    """
    Inverse transform a Johnson SU-normalized series back to its original distribution.
    
    Parameters:
    - transformed_series (pd.Series): The transformed (normalized) series.
    - jsu_params (tuple): The parameters of the Johnson SU distribution.
    
    Returns:
    - original_series (pd.Series): The series transformed back to its original distribution.
    """
    uniform_values = norm.cdf(transformed_series)
    original_values = johnsonsu.ppf(uniform_values, *jsu_params)
    original_series = pd.Series(original_values, index=transformed_series.index)

    return original_series


def tail_specific_transform(series, left_threshold, transformation='log'):
    """
    Applies a tail-specific transformation on the left tail of the series.
    
    Parameters:
    - series (pd.Series): Input series to transform.
    - left_threshold (float): The threshold below which the left tail transformation will be applied.
    - transformation (str): Type of transformation to apply ('log' for log-transform).
    
    Returns:
    - transformed_series (pd.Series): Transformed series with the left tail adjusted.
    """
    # Create a copy to avoid modifying the original data
    transformed_series = series.copy()
    
    # Apply a log transform to values in the left tail
    if transformation == 'log':
        left_tail = transformed_series < left_threshold
        # Transform with sign preservation
        transformed_series[left_tail] = np.log1p(np.abs(transformed_series[left_tail])) * np.sign(transformed_series[left_tail])
    
    return transformed_series


def tail_specific_inverse_transform(series, left_threshold, transformation='log'):
    """
    Inverse of the tail-specific log transformation applied to the left tail.
    
    Parameters:
    - series (pd.Series): Input transformed series.
    - left_threshold (float): The threshold below which the left tail was transformed.
    - transformation (str): The type of transformation applied ('log').
    
    Returns:
    - inverse_transformed_series (pd.Series): Series after reversing the log transformation on the left tail.
    """
    # Create a copy to avoid modifying the original data
    inverse_transformed_series = series.copy()
    transformed_left_threshold = np.expm1(left_threshold) * np.sign(left_threshold)
    transformed_left_threshold = np.log1p(np.abs(left_threshold))*np.sign(left_threshold)
    # val_neg = np.expm1(-left_threshold); print(val_neg)
    # val_pos = np.expm1(left_threshold); print(val_pos)

    # if val_pos>val_neg: 
    #     transformed_left_threshold = val_pos
    # else:
    #     transformed_left_threshold = -1*val_neg
    print(transformed_left_threshold)
    if transformation == 'log':
        left_tail = inverse_transformed_series < transformed_left_threshold
        # Inverse transform with sign preservation
        z = series[left_tail].copy()
        x1 = np.expm1(z)
        x2 = np.expm1(-z)
        x = np.empty_like(x1)
        x[x1>x2] = x1[x1>x2]
        x[x1<x2] = x2[x1<x2]
        print(np.shape(x))
        print(np.shape(inverse_transformed_series[left_tail]))
        # inverse_transformed_series[left_tail] = (np.expm1(np.abs(inverse_transformed_series[left_tail]))) * np.sign(inverse_transformed_series[left_tail])
        inverse_transformed_series[left_tail] = x
    
    return inverse_transformed_series


# This function will first undo the log transformation, then undo the Johnson SU transformation
# def full_inverse_transform(series, jsu_params, left_threshold):
#     """
#     Inverts the tail-specific log transformation and the Johnson SU transformation.
    
#     Parameters:
#     - series (pd.Series): Transformed series that needs to be inverted.
#     - jsu_params (tuple): Parameters from the Johnson SU distribution.
#     - left_threshold (float): The threshold for the tail-specific transformation.
    
#     Returns:
#     - original_series (pd.Series): Series restored to its original form.
#     """
#     # Step 1: Undo the tail-specific log transformation
#     series_after_tail_inverse = tail_specific_inverse_transform(series, left_threshold=left_threshold)
    
#     # Step 2: Undo the Johnson SU transformation
#     original_series = johnson_su_inverse_transform(series_after_tail_inverse, jsu_params)
    
#     return original_series


def ewm_rolling_beta(df: pd.DataFrame, column, ewm_alpha, min_periods, market_df):
    Y_series = df[column]
    X_series = market_df[column]
    XY_series = X_series * Y_series
    ewm_params = {'alpha': ewm_alpha, 'min_periods': min_periods}
    XY_mean = XY_series.ewm(**ewm_params).mean()
    X_mean = X_series.ewm(**ewm_params).mean()
    Y_mean = Y_series.ewm(**ewm_params).mean()
    X_variance = X_series.ewm(alpha=ewm_alpha).var()
    return (XY_mean - X_mean * Y_mean) / X_variance

# Grid search function for a single alpha value
def grid_search_alpha(alpha, train, market_df, left_threshold):
    result = {}
    
    # Compute beta_ewma using the rolling beta function
    train['beta_ewma'] = train.groupby('Code', group_keys=False).apply(ewm_rolling_beta, 'WinsorReturn', alpha, 45, market_df)
    
    # Transform beta_ewma using the Johnson SU transform function
    # train['beta_ewma_jsunorm'], jsu_params_ewma = johnson_su_transform(train['beta_ewma'])
    # train['beta_ewma_norm'] = tail_specific_transform(train['beta_ewma_jsunorm'], left_threshold=left_threshold)
    
    # Ensure that the holdout set isn't being used to optimize the hyperparameter
    X_ewma, y_baseline, X_ewma_holdout, y_holdout = training_dataset(train, feature_list=['beta_ewma'], target='beta_45target', holdout_year=2021)
    
    # Drop the holdout data
    del X_ewma_holdout, y_holdout
    
    # OLS Training
    X_ewma = sm.add_constant(X_ewma)
    ewma_model = sm.OLS(y_baseline, X_ewma).fit()
    train_predictions = ewma_model.predict(X_ewma)
    
    # Metrics
    rmse_train = mean_squared_error(y_baseline, train_predictions, squared=False)
    mae_train = mean_absolute_error(y_baseline, train_predictions)
    pearson_train = pearsonr(y_baseline, train_predictions)[0]
    spearman_train = spearmanr(y_baseline, train_predictions)[0]
    
    # Store Results
    result = {
        'alpha': alpha,
        'rmse_train': rmse_train,
        'mae_train': mae_train,
        'pearson_train': pearson_train,
        'spearman_train': spearman_train
    }
    
    return result

### FEATURE ENGINEERING 
def calc_volatility(_win, series, data:pd.DataFrame =None):
    if data ==None: data = train
    col_name = f'sigma_{_win}'
    data[col_name] = data.groupby('Code', group_keys=False)[series].apply(
        pandas_winsor).groupby('Code').rolling(_win).std().bfill().droplevel(0)
    return data
    # return data[col_name]



def calc_bbands(periods:list, series_col:str, data:pd.DataFrame = None):
    if data == None: data = train
    for p in periods:
        bbmid_col = f'sma_{p}'
        data[bbmid_col] = data.groupby('Code')[series_col].rolling(p).mean().droplevel(0)
        bb_std = data.groupby('Code')[series_col].rolling(p).std().droplevel(0)
        data[f'bbhigh_{p}'] = data[bbmid_col] + 2*bb_std
        data[f'bblow_{p}'] = data[bbmid_col] - 2*bb_std
        data[f'bbceil_{p}'] = 1-data[series_col]/data[f'bbhigh_{p}']
        data[f'bbfloor_{p}'] = 1-data[f'bblow_{p}']/data[series_col]
        # data[f'bbdiff_{p}'] = data[f'bbceil_{p}']+data[f'bbfloor_{p}']

### TRAINING

class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(self,
                 n_splits=3,
                 train_period_length=121,
                 test_period_length=21,
                 lookahead=None,
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values(1).unique()
        days = sorted(unique_dates, reverse=True)

        # days = unique_dates
        # days = sorted(unique_dates, reverse=False)

        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = X.reset_index()['Date']
        # self.dates = dates
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates > days[train_start])
                              & (dates <= days[train_end])].index
            test_idx = dates[(dates > days[test_start])
                             & (dates <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits        