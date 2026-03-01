from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
    )

    plt.xlabel("Anomaly score η")
    plt.ylabel("Health degradation (1−Ra)")

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("outputs/ra_degradation_vs_eta.png", bbox_inches="tight", pad_inches=0.01)
    plt.show()

def plot_ra_vs_eta_grid(df, col_wrap=5, height=1.8):
    df = df.copy()
    df["degradation"] = 1 - df["Ra"]

    endpoints = sorted(
        df["endpoint"].unique(),
        key=lambda x: int(x.split()[-1])
    )

    colors = sns.color_palette("colorblind", len(endpoints))
    palette = dict(zip(endpoints, colors))

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        rc={
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 13,
            "figure.dpi": 300,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.5,
        }
    )

    g = sns.FacetGrid(
        df,
        col="endpoint",
        col_order=endpoints,
        hue="endpoint",
        palette=palette,
        col_wrap=col_wrap,
        height=height,
        aspect=1.1,
        sharex=True,
        sharey=True,
        despine=False
    )

    g.map_dataframe(
        sns.scatterplot,
        x="eta",
        y="degradation",
        alpha=0.6,
        s=8,
        linewidth=0
    )

    # remove labels individuais
    for ax in g.axes.flat:
        ax.set_ylabel("")
        ax.set_xlabel("")

    # label global
    g.figure.text(
        0.5, -0.02,
        "Anomaly score η",
        ha="center",
        va="center"
    )

    g.figure.text(
        -0.01, 0.5,
        "Health degradation (1 − Ra)",
        ha="center",
        va="center",
        rotation="vertical"
    )

    g.set_titles("{col_name}")

    for ax in g.axes.flat:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.tight_layout(pad=0.3)
    plt.savefig("outputs/ra_degradation_vs_eta_grid.png", bbox_inches="tight", pad_inches=0.01)
    plt.show()

def plot_ra_vs_kalman_params_grid(df_k, df_Q, df_R, figsize=(7, 3.2)):
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif"
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        sharey=True
    )

    configs = [
        (df_k, "k", axes[0], "Gain k"),
        (df_Q, "Q", axes[1], "Process noise Q"),
        (df_R, "R", axes[2], "Measurement noise R"),
    ]

    legend_handles = None
    legend_labels = None

    for df, param, ax, title in configs:

        summary = (
            df
            .groupby(["endpoint", param])
            .Ra.mean()
            .reset_index()
        )

        line = sns.lineplot(
            data=summary,
            x=param,
            y="Ra",
            hue="endpoint",
            marker="o",
            ax=ax,
            legend=True
        )

        ax.set_title(title, fontsize=13)
        ax.set_xlabel(param)

        ax.grid(True, linestyle="--", alpha=0.4)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        # remove individual legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    axes[0].set_ylabel("Mean Rₐ")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=5,
        frameon=True,
        bbox_to_anchor=(0.5, -0.15),
        title="Endpoints"
    )

    sns.despine()
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("outputs/ra_vs_kalman_params_grid.png", bbox_inches="tight", pad_inches=0.01)
    plt.show()

def build_anomaly_summary_table(df):
    """
    Builds a table with anomaly frequency and mean duration (in windows)
    for each endpoint.

    Uses column:
        has_anomaly (0/1)
        window_id (ordering)
    """

    results = []

    for endpoint, group in df.groupby("endpoint"):
        group = group.sort_values("window_id").copy()
        total_windows = len(group)

        group["start"] = (
            (group["has_anomaly"] == 1) &
            (group["has_anomaly"].shift(1, fill_value=0) == 0)
        )

        group["event_id"] = group["start"].cumsum()
        anomaly_windows = group[group["has_anomaly"] == 1]

        if len(anomaly_windows) > 0:

            durations = (
                anomaly_windows
                .groupby("event_id")
                .size()
            )

            mean_duration = durations.mean()
            total_events = len(durations)

        else:
            mean_duration = 0.0
            total_events = 0

        frequency = total_events / total_windows

        results.append({
            "endpoint": endpoint,
            "anomaly_frequency": frequency,
            "mean_duration_windows": mean_duration,
            "total_events": total_events,
            "total_windows": total_windows
        })

    result = pd.DataFrame(results)

    # numerical order
    result = result.sort_values(
        "endpoint",
        key=lambda col: col.str.extract(r'(\d+)').astype(int)[0]
    )

    return result.reset_index(drop=True)

def plot_event_impact_grid(
    events_k,
    events_Q,
    events_R,
    events_window,
    figsize=(7,6),
):
    """
    Generate a 2x2 grid of boxplots showing impact distribution for:
    k, Q, R, window_size

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
    plt.savefig("outputs/ra_params_impact_grid.png", dpi=300, bbox_inches="tight")
    plt.show()

def build_summary_table(df, param_col):
    df = df.copy()
    df["degradation"] = 1 - df["Ra"]

    summary = df.groupby(param_col).agg(
        mean_Ra=("Ra", "mean"),
        std_Ra=("Ra", "std"),
        mean_degradation=("degradation", "mean"),
        std_degradation=("degradation", "std")

    ).reset_index()

    return summary

def build_event_summary(df,
                        time_col="window_start",
                        anomaly_col="has_anomaly",
                        Ra_col="Ra"):

    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(["endpoint", time_col])

    summaries = []

    for endpoint, group in data.groupby("endpoint"):
        group = group.copy()

        group["start"] = (
            (group[anomaly_col] == 1) &
            (group[anomaly_col].shift(1, fill_value=0) == 0)
        )

        group["event_id"] = group["start"].cumsum()

        anomaly_windows = group[group[anomaly_col] == 1]

        if anomaly_windows.empty:
            continue

        durations = anomaly_windows.groupby("event_id").size()

        mean_duration = durations.mean()
        total_events = len(durations)
        total_windows = len(group)

        frequency = total_events / total_windows

        mean_Ra = anomaly_windows[Ra_col].mean()

        summaries.append({
            "endpoint": endpoint,
            "mean_duration": mean_duration,
            "frequency": frequency,
            "mean_Ra": mean_Ra,
            "degradation": 1 - mean_Ra,
            "total_events": total_events
        })

    return pd.DataFrame(summaries)

def plot_classification_metrics_bar(
    metrics_df,
    window_size=50,
    figsize=(3.5, 2.6),
    subplot_label=None
):

    df = metrics_df.copy()
    df = df[df["window_size"] == window_size]

    endpoints = sorted(
        df["endpoint"].unique(),
        key=lambda x: int(x.split()[-1])
    )

    metrics = ["precision", "recall", "f1", "accuracy"]
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        rc={
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
        }
    )

    palette = {
        "precision": "#4C72B0",
        "recall": "#55A868",
        "f1": "#C44E52",
        "accuracy": "#8172B2"
    }

    fig, ax = plt.subplots(figsize=figsize)
    group_spacing = 1.6
    bar_width = 0.22

    x = np.arange(len(endpoints)) * group_spacing
    for i, metric in enumerate(metrics):
        values = [
            df[df["endpoint"] == ep][metric].values[0]
            for ep in endpoints
        ]

        offset = (i - 1.5) * bar_width
        linewidth = 1.0 if metric == "f1" else 0.4
        zorder = 3 if metric == "f1" else 2

        ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=palette[metric],
            edgecolor="black",
            linewidth=linewidth,
            label=metric,
            zorder=zorder
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        endpoints,
        rotation=35,
        ha="right"
    )

    ax.set_ylabel("Metric value")
   # ax.set_title(
   #     f"Window size = {window_size} s",
   #     fontsize=9,
   #     pad=4
   # )

    ax.set_ylim(0, 1.05)
    ax.legend(
        ncol=4,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        columnspacing=1.2,
        handlelength=1.8
    )

    if subplot_label is not None:
        ax.text(
            0.01,
            0.98,
            subplot_label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top"
        )

    sns.despine(ax=ax)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", visible=False)

    plt.tight_layout(pad=0.6)
    plt.savefig("outputs/ra_classification_metrics_bar.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_anomaly_timeline_seaborn(
    res_df,
    time_col="window_start",
    window_size_sec=30,
    figsize=(10, 4),
    save_path=None
):
    """
    Plots anomaly regime timeline using matplotlib/seaborn.
    Each anomaly event is shown as a horizontal bar.
    """

    df = res_df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["has_anomaly"] = df["has_anomaly"].astype(int)
    df = df.sort_values(["endpoint", time_col])

    endpoints = sorted(
        df["endpoint"].unique(),
        key=lambda x: int(x.split()[-1])
    )

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        rc={
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = {endpoint: i for i, endpoint in enumerate(endpoints)}
    total_blocks = 0

    for endpoint in endpoints:
        sub = df[df["endpoint"] == endpoint].copy()
        sub["start"] = (
            (sub["has_anomaly"] == 1) &
            (sub["has_anomaly"].shift(1, fill_value=0) == 0)
        )
        sub["event_id"] = sub["start"].cumsum()
        events = sub[sub["has_anomaly"] == 1].groupby("event_id")

        y = y_positions[endpoint]
        for _, event in events:

            start = event[time_col].iloc[0]
            end = event[time_col].iloc[-1] + pd.Timedelta(seconds=window_size_sec)

            ax.barh(
                y=y,
                width=end - start,
                left=start,
                height=0.6,
                color="red",
                edgecolor="black",
                linewidth=0.3
            )

            total_blocks += 1

    # configurar eixo Y
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))

    ax.set_xlabel("Time")
    ax.set_ylabel("Endpoint")
    ax.set_title("Persistent anomaly regimes across endpoints")

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            bbox_inches="tight"
        )
    plt.show()
    print(f"Rendered {total_blocks} anomaly blocks")

def _define_plot_limits(df, time_col):
    anomaly_col = 'has_anomaly'
    df["start_flag"] = (
        (df[anomaly_col] == 1) &
        (df[anomaly_col].shift(1, fill_value=0) == 0)
    )

    df["event_id"] = df["start_flag"].cumsum()
    anomaly_windows = df[df[anomaly_col] == 1]
    durations = anomaly_windows.groupby("event_id").size()

    max_event_id = durations.idxmax()
    event = anomaly_windows[anomaly_windows["event_id"] == max_event_id]
    start_time = event[time_col].iloc[0]

    #first_anomaly = df.loc[df['has_anomaly'] == 1, time_col].min()
    begin_plot = start_time - pd.Timedelta(minutes=10)
    end_plot = start_time + pd.Timedelta(minutes=60)
    print(f'greatest anomaly at:{start_time} - begin plot:{begin_plot} - end plot:{end_plot}')
    return begin_plot, end_plot

def _normalize_traffic_flow(df):
    flow_log = np.log10(df['total_requests'] + 1)
    flow_scaled = (
        (flow_log - flow_log.min())/
        (flow_log.max() - flow_log.min() + 1e-9)
    )
    return flow_scaled


def _add_anomaly_shading(df, time_col, fig):
    df = df.copy()
    gt_df = df[df["has_anomaly"] == "Attack"]

    fig.add_trace(
        go.Scatter(
            x=gt_df[time_col],
            y=[0.01] * len(gt_df),  # base do gráfico
            mode="markers",
            marker=dict(
                symbol="line-ns-open",  # linha vertical pequena
                size=5,
                color="red",
                line=dict(width=2)
            ),
            name="Anomaly",
            showlegend=True,
            legendgroup="ground_truth",
            legendgrouptitle_text="Ground truth"
        ),
        row=2, col=1
    )

    for t in gt_df[time_col]:
        fig.add_vline(
            x=t,
            line_width=1,
            line_dash="dot",
            line_color="red",
            row="all", col=1
        )

def _plot_Ra(df, time_col, fig, row):
    color_map = {'Benign': 'green', 'Attack': '#d62728'}

    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df["eta"],
            mode='lines',
            line=dict(color="orange", width=1.5),
            name="$\\eta$",
            showlegend=True,
            legendgroup="detection",
            legendgrouptitle_text ="Detection",
        ),
        row=row, col=1
    )

    md_df = df.copy()
    fig.add_trace(
        go.Scatter(
            x=md_df[time_col],
            y=md_df['Ra'],
            mode='lines',
            line=dict(color="#1f77b4", width=2, dash="solid"),
            name='$R_a$',
            hovertemplate='<br>Time: %{x}<br>Ra: %{y:.3f}<br>Status: %{marker.color}<extra></extra>',
            showlegend=True,
            legendgroup="detection",
        ),
        row=row,
        col=1
    )


def _plot_metrics(df, time_col, fig, row):
    metric_colors = {
        #'D': '#1f77b4',
        #'Z': '#ff7f0e',
        #'Delta': '#2ca02c',
        #'Ra2': "#fd61bc",
        #'C2': "#ff0000",
        'fDp': "#19D3F3",
        'fDeltap': "#AB63FA",
        'fZp': "#FF6692"
    }

    metric_names = {
        "fDeltap":"$f'_\\Delta$",
        "fDp": "$f'_D$",
        "fZp": "$f'_Z$"
    }

    for metric, color in metric_colors.items():
        fig.add_trace(
            go.Scatter(
                x=df[time_col],
                y=df[metric],
                mode='lines',
                line=dict(color=color, width=2),
                name=metric_names.get(metric),
                legendgroup="features",
                showlegend=True
            ),
            row=row,
            col=1
        )

def _plot_traffic(df, time_col, fig, row):
    flow_scaled = _normalize_traffic_flow(df)
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=flow_scaled,
            mode='lines',
            name='log traffic',
            line=dict(color="#8E8E8E", width=1.2, dash="solid"),#00c3ff
            opacity=0.5,
            hovertemplate='Time: %{x}<br>Request: %{y:.3f}<extra></extra>',
            showlegend=True,
            legendgroup="features",
            legendgrouptitle_text="Observables",
        ),
        row=row,
        col=1
    )

def plot_results(res_df,df, endpoint):
    #unique_pairs = res_df['endpoint'].drop_duplicates().sample(min(plot_examples, len(res_df)))

    time_col = 'window_start'
    sub_res = res_df[(res_df['endpoint'] == endpoint)].sort_values(time_col).copy()
    df_flow = df[(df['endpoint'] == endpoint)].sort_values('time_local').copy()

    begin_plot, end_plot = _define_plot_limits(sub_res, time_col)

    sub_res = sub_res[(sub_res[time_col] >= begin_plot) & (sub_res[time_col] <= end_plot)].copy()
    sub_res[time_col] = sub_res[time_col] + pd.Timedelta(seconds=30)
    sub_res['has_anomaly'] = sub_res['has_anomaly'].map({1: 'Attack', 0: 'Benign'})

    df_flow = df_flow[(df_flow['time_local'] >= begin_plot) & (df_flow['time_local'] <= end_plot)].copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.5],
    )

    _plot_traffic(df_flow, 'time_local', fig, row=1)
    _plot_metrics(sub_res, time_col, fig, row=1)
    _plot_Ra(sub_res, time_col, fig, row=2)
    _add_anomaly_shading(sub_res, time_col, fig)

    # --- Layout ---
    fig.update_layout(
        #title=f'Metrics evolution for the {endpoint}',
        template='plotly_white',
        #legend_title_text='Metric',
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(
            font=dict(size=17),
            grouptitlefont=dict(size=18),
            tracegroupgap=35,
        ),
    )
    fig.update_yaxes(title_text="Traffic features", title_font=dict(size=18), row=1, col=1, showgrid=True)
    fig.update_xaxes(row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text="Health vs Anomaly",title_font=dict(size=18), row=2, col=1, showgrid=True)
    fig.update_xaxes(title_text="Time", title_font=dict(size=18), tickfont=dict(size=14), row=2, col=1, showgrid=True)

    fig.write_image(f"outputs/metrics_by_time_{endpoint.replace(' ', '')}.png")
    fig.show()

def plot_regimes_clusters(summary_df, distance_threshold=0.08,  desired_max_size=40, radius_offset=5):
    df = summary_df.copy()
    x = df["mean_duration"].values
    y = df["frequency"].values

    log_x = np.log10(x)
    log_y = np.log10(y)

    sizeref = 2. * df["total_events"].max() / (desired_max_size ** 2)
    x_log_min, x_log_max = log_x.min(), log_x.max()
    y_log_min, y_log_max = log_y.min(), log_y.max()

    fig_width_px = 650
    fig_height_px = 480
    margin_l, margin_r, margin_t, margin_b = 70, 90, 30, 60
    axis_width_px = fig_width_px - margin_l - margin_r
    axis_height_px = fig_height_px - margin_t - margin_b

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=df["total_events"],
            sizemode="area",
            sizeref=sizeref,
            sizemin=6,
            color=df["degradation"],
            colorscale="RdBu_r",
            line=dict(width=1, color="black"),
            colorbar=dict(
                title=dict(
                    text="Health degradation (1 − Rₐ)",
                    side="right",
                ),
                thickness=18,
                len=0.75,
            )
        ),
        text=df["endpoint"],
        showlegend=False
    ))

    used = set()
    label_x, label_y, label_text = [], [], []
    for i in range(len(df)):
        if i in used:
            continue

        group = [i]
        for j in range(i+1, len(df)):
            dist = np.sqrt(
                (log_x[i] - log_x[j])**2 +
                (log_y[i] - log_y[j])**2
            )
            if dist < distance_threshold:
                group.append(j)
                used.add(j)

        gx = np.mean(x[group])
        gy = np.mean(y[group])

        labels = [
            df.iloc[k]["endpoint"].split()[-1]
            for k in group
        ]

        # calculates radius
        max_events = df.iloc[group]["total_events"].max()
        diameter_px = np.sqrt(max_events / sizeref)
        radius_px = diameter_px / 2

        # converts radius → log coord
        radius_log_x = (radius_px + radius_offset) / axis_width_px * (x_log_max - x_log_min)
        radius_log_y = (radius_px + radius_offset) / axis_height_px * (y_log_max - y_log_min)

        label_x.append(10**(np.log10(gx) + radius_log_x))
        label_y.append(10**(np.log10(gy) + radius_log_y))
        label_text.append(",".join(labels))

    fig.add_trace(go.Scatter(
        x=label_x,
        y=label_y,
        mode="text",
        text=label_text,
        textposition="middle right",
        textfont=dict(size=14, family="Serif", color="black"),
        showlegend=False
    ))

    fig.update_layout(
        template="simple_white",
        width=fig_width_px,
        height=fig_height_px,
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        xaxis=dict(
            title="Mean anomaly duration (windows)",
            type="log",
            tickvals=[1,10,100,1000],
            ticktext=["10⁰","10¹","10²","10³"]
        ),
        yaxis=dict(
            title="Anomaly frequency",
            type="log",
            exponentformat="power",
            showexponent="all"
        ),
        font=dict(family="Serif", size=16)
    )
    fig.write_image("outputs/ra_endpoints_cluster.png")
    fig.show()