import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

# Glbal settings
OUTPUT_PATH = "outputs/"
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 9, "figure.dpi": 300})

def compute_classification_metrics(df_eta):
    """
    Calculates classification metrics grouped by endpoint and window size
    Args:
        df_eta (DataFrame): raw eta dataset
    Returns:
        DataFrame: ROC AUC and PR AUC by endpoint and window size
    """
    metrics = (
        df_eta.groupby(["endpoint", "window_size"])
        .apply(lambda g: pd.Series({
            "ROC_AUC": roc_auc_score(g["has_anomaly"], g["eta"]),
            "PR_AUC": average_precision_score(g["has_anomaly"], g["eta"])
        }), include_groups=False)
        .reset_index()
    )
    metrics.to_csv(f"{OUTPUT_PATH}eta_metrics.csv", index=False)
    return metrics


def get_summary_stats(df_eta_metrics, group_by=None):
    """
    Generate description stats (Mean, Std, Min, Max)
    Args:
        df_eta_metrics (DataFrame): eta PR AUC and ROC AUC by endpoint
                                    and window size
        group_by (list): columns to consider when grouping
    Returns:
        DataFrame: grouped aggregated metrics
    """
    cols = ["ROC_AUC", "PR_AUC"]
    if group_by:
        return (df_eta_metrics.groupby(group_by)[cols]
                .agg(["mean", "std", "min", "max"])
                .reset_index())

    summary = (df_eta_metrics[cols]
               .agg(["mean", "std", "min", "max"]).T)
    return summary.rename(columns=str.capitalize)


def _prepare_tidy_data(df, x_col, metrics_map):
    """
    Helper to transform data into long format
    Args:
        df (DataFrame): eta dataset
        x_col (str): column to consider when melting
        metrics_map (dict): mapping of metric name to metric value
    Returns:
        DataFrame: long format data
    """
    df_plot = df.rename(columns=metrics_map)
    return df_plot.melt(
        id_vars=x_col,
        value_vars=list(metrics_map.values()),
        var_name="Metric",
        value_name="Score"
    )


def plot_detection_capability(df_metrics, figsize=(10, 5)):
    """
    Scatterplot comparing metrics per endpoint
    Args:
        df_metrics (DataFrame): raw eta dataset
        figsize (tuple): figure size
    """
    stats = (df_metrics
             .groupby("endpoint")[["PR_AUC", "ROC_AUC"]]
             .mean()
             .reset_index())
    stats = stats.sort_values("PR_AUC")

    df_melted = _prepare_tidy_data(stats, "endpoint", {
        "PR_AUC": "mean PR AUC",
        "ROC_AUC": "mean ROC AUC"
    })

    plt.figure(figsize=figsize)
    ax = sns.lineplot(
        data=df_melted, x="endpoint", y="Score",
        hue="Metric", style="Metric", markers=True, markersize=8, linewidth=2
    )

    ax.set(title="Detection Capability Across Endpoints", xlabel="Endpoint", ylabel="Score", ylim=(0, 1.05))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}detection_capability_sns.png")
    plt.show()


def plot_combined_analysis(df_metrics, df_raw, figsize=(8, 6)):
    """
    Combined plot for window size analysis: eta boxplot and perfomance lineplot
    Args:
        df_metrics (DataFrame): eta metrics dataset
        df_raw (DataFrame): raw eta dataset
        figsize (tuple): figure size
    """
    df_raw['Label'] = df_raw['has_anomaly'].map({0: 'Normal', 1: 'Anomaly'})

    window_stats = df_metrics.groupby("window_size")[["PR_AUC", "ROC_AUC"]].mean().reset_index()
    window_stats = window_stats.sort_values("PR_AUC")
    df_melted = _prepare_tidy_data(window_stats, "window_size", {
        "PR_AUC": "mean PR AUC",
        "ROC_AUC": "mean ROC AUC"
    })

    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]})

    # Subplot 0: Boxplot
    sns.boxplot(data=df_raw, x="window_size", y="eta", hue="Label", width=0.6, ax=axes[0])
    axes[0].set(xlabel="", ylabel="$\eta$")

    # Subplot 1: Lineplot
    sns.lineplot(
        data=df_melted, x="window_size", y="Score",
        hue="Metric", style="Metric", markers=True, dashes=True, markersize=8, ax=axes[1]
    )
    axes[1].set(xlabel="Window Size (s)", ylabel="Score", ylim=(0, 1.05))
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}eta_analysis_combined.png")
    plt.show()