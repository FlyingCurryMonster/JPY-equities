import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import probplot
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from scipy.stats import pearsonr


def plot_qq(series, dist="norm", title="Q-Q Plot", dist_params=None):
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    clean_series = series.dropna()

    # Shift and scale data to Z score
    if dist == "norm":
        standardized_series = (clean_series - clean_series.mean())
        standardized_series /= clean_series.std()

    # Otherwise rescale by given parameters and
    # distribution family
    else:
        if dist_params is None:
            raise ValueError(
                "Distribution parameters (dist_params)" +
                f"must be provided for {dist}."
            )
        cdf_values = stats.__dict__[dist].cdf(clean_series, *dist_params)
        standardized_series = stats.norm.ppf(cdf_values)

    plt.figure()
    (theoretical_quantiles, ordered_values), regression_line = probplot(
        standardized_series, dist=dist, sparams=dist_params
    )
    plt.plot(theoretical_quantiles, ordered_values, "o", label="Data Points")
    plt.plot(
        theoretical_quantiles,
        regression_line[0] * theoretical_quantiles + regression_line[1],
        "r-",
        label="Best Fit Line",
    )
    plt.plot(theoretical_quantiles, theoretical_quantiles, "k--", label="y=x")
    plt.title(title)
    plt.xlabel(f"Theoretical Quantiles of {dist} distribution")
    plt.ylabel(f"Ordered Values from {clean_series.name}")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_actual_vs_predicted(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    model_name="Model",
    actual_col="actual_target",
    predicted_col="target_prediction",
):
    y_train_actual = train_df[actual_col]
    y_train_pred = train_df[predicted_col]
    pearson_train = pearsonr(y_train_actual, y_train_pred)[0]
    rmse_train = root_mean_squared_error(y_train_actual, y_train_pred)
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)

    y_holdout_actual = holdout_df[actual_col]
    y_holdout_pred = holdout_df[predicted_col]
    pearson_holdout = pearsonr(y_holdout_actual, y_holdout_pred)[0]
    rmse_holdout = root_mean_squared_error(y_holdout_actual, y_holdout_pred)
    mae_holdout = mean_absolute_error(y_holdout_actual, y_holdout_pred)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    axa, axb = ax

    sns.scatterplot(x=y_train_pred, y=y_train_actual, alpha=0.3, ax=axa)
    axa.set_title(
        f"TRAINING SET\n{model_name}\nPearson r={pearson_train:.3g}\n"
        + f"RMSE: {rmse_train:.3f}\nMAE: {mae_train:.3f}",
        fontsize=14,
    )
    axa.set_xlabel("Predicted Values")
    axa.set_ylabel("Actual Values")

    sns.scatterplot(x=y_holdout_pred, y=y_holdout_actual, alpha=0.3, ax=axb)
    axb.set_title(
        f"HOLDOUT SET\n{model_name}\nPearson r={pearson_holdout:.3g}\n"
        + f"RMSE: {rmse_holdout:.3f}\nMAE: {mae_holdout:.3f}",
        fontsize=14,
    )
    axb.set_xlabel("Predicted Values")
    axb.set_ylabel("Actual Values")

    for a in [axa, axb]:
        a.set_xlim(a.get_xlim())
        a.set_ylim(a.get_ylim())
        a.plot(a.get_xlim(), a.get_xlim(),
               "--", color="tab:orange", linewidth=3, label="y = x")
    axa.legend(fontsize=12)

    print("Training Set Metrics:")
    print(f"Pearson r: {pearson_train:.3f}")
    print(f"RMSE: {rmse_train:.3f}")
    print(f"MAE: {mae_train:.3f}\n")

    print("Holdout Set Metrics:")
    print(f"Pearson r: {pearson_holdout:.3f}")
    print(f"RMSE: {rmse_holdout:.3f}")
    print(f"MAE: {mae_holdout:.3f}")


def plot_time_metrics(
        baseline_metrics_df, metrics_df: pd.DataFrame, model_name: str):
    assert (baseline_metrics_df.columns == metrics_df.columns).all()
    columns = metrics_df.columns
    n = len(columns)
    fig, ax = plt.subplots(n, 1, figsize=(9.2, 5.6 * n * 2 / 3))
    fig.subplots_adjust(hspace=0.5)
    for i, col in enumerate(columns):
        metric_mean = metrics_df[col].mean()
        baseline_metric_mean = baseline_metrics_df[col].mean()
        ax[i].set_title(
            f"Mean daily baseline {col} = {baseline_metric_mean:.3g}\n"
            + f"Mean daily {model_name} {col} = {metric_mean:.3g}"
        )
        ax[i].axhline(baseline_metric_mean, color="tab:blue", linestyle="--")
        ax[i].axhline(metric_mean, color="tab:orange", linestyle="--")
        sns.lineplot(
            baseline_metrics_df, x=metrics_df.index.name, y=col,
            ax=ax[i], marker="o", label="baseline"
        )
        sns.lineplot(metrics_df, x=metrics_df.index.name, y=col,
                     ax=ax[i], marker="o", label=model_name)
        ax[i].legend()


# def feature_timeseries_plot(
# tickers: list, feature, target, df: pd.DataFrame = None):
#     if df is None:
#         df = train

#     fig, ax = plt.subplots(2, len(tickers), figsize=(14, 10))

#     for i, tick in enumerate(tickers):
#         ax[0, i].set_title(f"Security code {tick}")
#         ax[1, i].set_title(
#               f"beta rank {universe_stats.loc[tick, 'beta_rank']}")
#         sns.lineplot(df.loc[tick], x="Date", y=target, ax=ax[0, i])
#         sns.lineplot(df.loc[tick], x="Date", y=feature, ax=ax[1, i])


# def feature_scatter(feature, target,
# spearman=False, df: pd.DataFrame = None, fig=None, ax=None):
#     if df is None:
#         df = train.loc[train.index.get_level_values(0) != 1320]
#     if ax is None:
#         fig, ax = plt.subplots()
#     df = df[[feature, target]].dropna()
#     sns.scatterplot(data=df, x=feature, y=target, ax=ax)
#     pearson_r = pearsonr(df[feature], df[target])[0]
#     ax.set_title(f"{feature} vs {target} Pearson r={pearson_r:.3g}")
#     if spearman:
#         spearman_r = spearmanr(df[feature], df[target])[0]
#         print(f"{feature} vs {target} Spearman r={spearman_r:.3g}")


# def feature_heatmap(
#   feat_list: list, targets: list = None, data: pd.DataFrame = None):
#     if data is None:
#         data = train
#     if targets is None:
#         targets = ["beta_target_45"]
#     cols = feat_list + targets
#     fig, ax = plt.subplots(
#       figsize=(1.5 * len(feat_list), 1.5 * len(feat_list)))
#     sns.heatmap(data[cols].dropna().corr(), ax=ax,
#       cmap="viridis", annot=True, fmt=".2g")
