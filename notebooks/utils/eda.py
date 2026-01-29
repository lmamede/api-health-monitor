import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates
from scipy.stats import poisson, kstest
import numpy as np

class EDA:
    def __init__(self, df):
        self.counts = None
        self.df = df

    def plot_global_request_rate(self, frequency):
        """ Group df based on the frequency provided"""
        time_col = 'time_local'
        label_col = 'type'

        counts = (self.df.groupby([label_col, pd.Grouper(key=time_col, freq=frequency)])
                  .size()
                  .reset_index(name='count'))

        # Plot evolution over time
        fig = px.line(counts, x=time_col, y='count', color=label_col,
                      title='Request frequency over time',
                      labels={time_col: 'Time', 'count': 'Count'})
        fig.show()

    def set_time_axis_30min(self, ax, interval):
        """
        Force a fixed 30-minute granularity on the x-axis,
        independent of the data resampling frequency.
        """
        locator = mdates.MinuteLocator(interval=interval)
        formatter = mdates.DateFormatter("%d %H:%M")

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    def plot_endpoint_request_rate(self, top_uris=[], frequency='min', top=6, window_size=None, start_offset_s=0, interval=30):
        time_col = 'time_local'
        label_col = 'type'  # e.g., normal / dos
        endpoint_col = 'endpoint'

        if not top_uris:
            top_uris = (
                self.df[endpoint_col]
                .value_counts()
                .head(top)
                .index
            )

        df = self.df[self.df[endpoint_col].isin(top_uris)].copy()
        first_day = df[time_col].dt.normalize().min()
        df = df[df[time_col].dt.normalize() == first_day].copy()
        df = df.set_index(time_col)

        counts = (
            df.groupby([endpoint_col, label_col])
            .resample(frequency, include_groups=False)
            .size()
            .fillna(0)
            .reset_index(name='count')
        )

        plot_start = counts['time_local'].iloc[0] + pd.to_timedelta(start_offset_s, unit='s')
        print('plot_start:', plot_start)
        counts = counts[counts[time_col] >= plot_start].copy()
        self.counts= counts

        figsize_plot = (15, 5)
        endpoints = counts[endpoint_col].unique()
        n_endpoints = len(endpoints)
        n_rows = 2
        n_cols = math.ceil(n_endpoints / n_rows)
        sharex=False
        interval = interval

        if len(top_uris) > 1:
            sharex=True
            n_cols = 3
            interval = 300
            n_rows = math.ceil(n_endpoints / n_cols)
            figsize_plot = (5 * n_cols, 3.5 * n_rows)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize_plot,
            sharex=sharex,
            sharey=False
        )

        if n_cols > 1:
            axes = axes.flatten()

        for ax, endpoint in zip(axes, endpoints):
            df_ep = counts[counts[endpoint_col] == endpoint]

            for label, df_lbl in df_ep.groupby(label_col):
                ax.plot(
                    df_lbl[time_col],
                    df_lbl["count"],
                    label=label
                )

            if window_size is not None:
                t_start = df_ep[time_col].min().floor('s')
                t_end = df_ep[time_col].max()

                window_bounds = pd.date_range(
                    start=t_start,
                    end=t_end,
                    freq=f'{window_size}'
                )

                for t in window_bounds:
                    ax.axvline(
                        t,
                        linestyle='-',
                        linewidth=1,
                        alpha=0.3,
                        color='red'
                    )

            self.set_time_axis_30min(ax, interval=interval)
            ax.tick_params(axis="x", labelrotation=80)
            ax.set_title(f"Endpoint: {endpoint}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Requests")
            ax.legend()

        # Remove empty subplots
        for ax in axes[len(endpoints):]:
            ax.remove()

        complement = f'\nwindow size {window_size}' if window_size is not None else ''
        fig.suptitle(
            f"Request frequency over time per endpoint{complement}",
            fontsize=14
        )

        plt.tight_layout(rect=(0., 0., 1., 0.95))
        plt.show()

    def poisson_window_metrics(self, sample):
        window_lam = sample.mean()
        window_var = sample.var()

        # KS test against Poisson(lambda)
        D, p = kstest(sample, poisson(window_lam).cdf)

        return pd.Series({
            "lambda": window_lam,
            "variance": window_var,
            "var_mean_ratio": window_var / window_lam if window_lam > 0 else np.nan,
            "ks_stat": D,
            "ks_pvalue": p,
            "n_samples": len(sample)
        })