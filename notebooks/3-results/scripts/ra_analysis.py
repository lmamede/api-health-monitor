from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_ra_vs_param_per_endpoint(df, param_name, figsize=(4,3)):
    summary = (
        df
        .groupby(["endpoint", param_name])
        .Ra.mean()
        .reset_index()
    )
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=summary,
        x=param_name,
        y="Ra",
        hue="endpoint",
        marker="o",
        legend=False
    )

    plt.xlabel(f"Sensitivity parameter {param_name}")
    plt.ylabel("Mean Ra")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_ra_vs_eta(df, figsize=(5,4)):
    df = df.copy()
    df["degradation"] = 1 - df["Ra"]

    plt.figure(figsize=figsize)

    ax = sns.scatterplot(
        data=df,
        x="eta",
        y="degradation",
        hue="endpoint",
        alpha=0.3,
        legend=True
    )

    sns.move_legend(
        ax, "upper center",
        bbox_to_anchor=(.5, -0.25),
        ncol=5,
        title="Endpoints",
        fontsize='small',
    )

    plt.xlabel("Anomaly score η")
    plt.ylabel("Health degradation (1−Ra)")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def compute_response_energy(df):
    df = df.copy()
    df["degradation"] = 1 - df["Ra"]

    energy = (
        df[df.has_anomaly==1]
        .groupby("endpoint")
        .degradation.sum()
        .reset_index(name="energy")
    )

    return energy

def extract_event_impacts(
    df,
    endpoint_col="endpoint",
    time_col="window_start",
    Ra_col="Ra",
    eta_col="eta",
    anomaly_col="has_anomaly",
    k_col="k",
    Q_col="Q",
    R_col="R",
    window_col="window_size",
):
    """
    Extract event-level Ra response metrics.
    Args:
        df (DataFrame)
        endpoint
        window_start
        Ra
        eta
        has_anomaly
        kf_k
        kf_Q
        kf_R
        window_size
    Returns:
         DataFrame: with one row per anomaly event.
    """
    df = df.copy()
    df = df.sort_values([endpoint_col, time_col])

    # degradation signal
    df["degradation"] = 1 - df[Ra_col]
    results = []

    for endpoint in df[endpoint_col].unique():
        sub = df[df[endpoint_col] == endpoint].copy()
        sub = sub.sort_values(time_col)

        # detect anomaly event boundaries
        sub["event_id"] = (
            (sub[anomaly_col] != sub[anomaly_col].shift())
            .cumsum()
        )

        # keep only anomaly events
        anomaly_events = sub[sub[anomaly_col] == 1]

        for event_id, event in anomaly_events.groupby("event_id"):
            event_start = event[time_col].iloc[0]
            event_end   = event[time_col].iloc[-1]

            # find baseline before event
            before = sub[sub[time_col] < event_start]

            if len(before) == 0:
                continue

            baseline_Ra = before[Ra_col].iloc[-1]
            min_Ra = event[Ra_col].min()
            impact = baseline_Ra - min_Ra

            # duration (number of windows)
            duration = len(event)

            # energy (integral of degradation)
            energy = event["degradation"].sum()

            # eta stats
            mean_eta = event[eta_col].mean()
            max_eta  = event[eta_col].max()

            # recovery time
            after = sub[sub[time_col] > event_end]
            recovery_time = np.nan

            if len(after) > 0:
                recovered = after[
                    after[Ra_col] >= baseline_Ra * 0.95
                ]

                if len(recovered) > 0:
                    recovery_time = (
                        recovered[time_col].iloc[0]
                        - event_end
                    )

                    recovery_time = recovery_time.total_seconds()

            # extract KF params safely
            k_val = event[k_col].iloc[0] if k_col in event else np.nan
            Q_val = event[Q_col].iloc[0] if Q_col in event else np.nan
            R_val = event[R_col].iloc[0] if R_col in event else np.nan
            w_val = event[window_col].iloc[0] if window_col in event else np.nan

            results.append({
                "endpoint": endpoint,
                "event_start": event_start,
                "event_end": event_end,
                "impact": impact,
                "duration": duration,
                "energy": energy,
                "baseline_Ra": baseline_Ra,
                "min_Ra": min_Ra,
                "mean_eta": mean_eta,
                "max_eta": max_eta,
                "recovery_time_s": recovery_time,
                "kf_k": k_val,
                "kf_Q": Q_val,
                "kf_R": R_val,
                "window_size": w_val
            })

    return pd.DataFrame(results)

def plot_response_vs_duration(df):
    df = df.copy()
    df["degradation"] = 1 - df["Ra"]

    events = extract_event_impacts(df)

    plt.scatter(
        events["duration"],
        events["impact"]
    )

    plt.xlabel("Event duration")
    plt.ylabel("Impact")
    plt.show()

def plot_event_impact_grid(
    events_k,
    events_Q,
    events_R,
    events_window,
    figsize=(7,6),
):
    """
    Generate a 2x2 grid of boxplots showing impact distribution for:
    k
    Q
    R
    window_size

    Each input must be output of extract_event_impacts().
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    sns.boxplot(
        data=events_k,
        x="kf_k",
        y="impact",
        ax=axes[0,0]
    )

    axes[0,0].set_title("Impact vs sensitivity (k)")
    axes[0,0].set_xlabel("k")
    axes[0,0].set_ylabel("Impact (ΔRa)")

    sns.boxplot(
        data=events_Q,
        x="kf_Q",
        y="impact",
        ax=axes[0,1]
    )

    axes[0,1].set_title("Impact vs process noise (Q)")
    axes[0,1].set_xlabel("Q")
    axes[0,1].set_ylabel("")

    sns.boxplot(
        data=events_R,
        x="kf_R",
        y="impact",
        ax=axes[1,0]
    )

    axes[1,0].set_title("Impact vs observation noise (R)")
    axes[1,0].set_xlabel("R")
    axes[1,0].set_ylabel("Impact (ΔRa)")

    sns.boxplot(
        data=events_window,
        x="window_size",
        y="impact",
        ax=axes[1,1]
    )

    axes[1,1].set_title("Impact vs window size")
    axes[1,1].set_xlabel("Window size (s)")
    axes[1,1].set_ylabel("")

    plt.tight_layout()
    plt.savefig("outputs/impact_grid_ieee.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_sensitivity_curve(
    df,
    param_col,
    figsize=(3.5,2.5)
):
    """
    param_col ∈ {"kf_k", "kf_Q", "kf_R", "window_size"}
    """
    df = df.copy()
    df["degradation"] = 1 - df["Ra"]

    summary = df.groupby(param_col)["degradation"].mean().reset_index()

    plt.figure(figsize=figsize)
    sns.lineplot(
        data=summary,
        x=param_col,
        y="degradation",
        marker="o"
    )

    plt.xlabel(param_col)
    plt.ylabel("Mean degradation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def build_summary_table(df, param_col):
    df = df.copy()
    df["degradation"] = 1 - df["Ra"]

    summary = df.groupby(param_col).agg(
        mean_Ra=("Ra","mean"),
        std_Ra=("Ra","std"),
        mean_degradation=("degradation","mean"),
        std_degradation=("degradation","std")

    ).reset_index()

    return summary

def compute_ra_classification_metrics(df, threshold=0.99):
    results = []
    grouped = df.groupby(["endpoint", "window_size"])

    for (endpoint, window_size), g in grouped:
        y_true = g["has_anomaly"]
        y_pred = (g["Ra"] < threshold)

        results.append({
            "endpoint": endpoint,
            "window_size": window_size,
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred)
        })

    return pd.DataFrame(results)
