import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import math
import matplotlib.dates as mdates
from scipy.stats import poisson, kstest
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

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

    def index_windows(self, window_indexed_df):
        window_indexed_df['window_id'] = (
                window_indexed_df['window_start']
                .astype(int)
                .rank(method='dense')
                .astype(int) - 1
        )

        window_indexed_df = window_indexed_df[['endpoint', 'window_id', 'time_local', 'count', 'is_anomaly']]

        window_indexed_df = window_indexed_df.rename(columns={
            'count': 'total_requests'
        })

        return window_indexed_df

    def poisson_evaluation_per_window(self, endpoint_df, window_size):
        endpoint_df['window_start'] = endpoint_df['time_local'].dt.floor(window_size).copy()
        endpoint_df = self.index_windows(endpoint_df)

        window_stats = (
            endpoint_df.groupby(["endpoint", "window_id"])["total_requests"]
            .apply(self.poisson_window_metrics)
            .unstack()
            .reset_index()
        )

        window_stats['poisson_deviation'] = (
                np.abs(window_stats['var_mean_ratio'] - 1) * window_stats['ks_stat']
        )

        window_stats['poisson_deviation_log'] = np.log1p(window_stats['poisson_deviation'])

        window_stats['flag_non_poisson'] = (
                (window_stats['var_mean_ratio'] > 1.5) |
                (window_stats['ks_pvalue'] < 0.01)
        )

        return window_stats

    def poisson_endpoint_metric(self, sample):
        endpoint_var_mean = sample.median(skipna=True)

        return pd.Series({
            "var_mean_ratio_median": endpoint_var_mean,
        })

    def poisson_evaluation_per_endpoint(self, all_endpoint_df, window_sizes):
        final_score = None

        for window_size in window_sizes:
            all_endpoint_df['window_start'] = all_endpoint_df['time_local'].dt.floor(window_size).copy()
            endpoint_df = self.index_windows(all_endpoint_df)

            window_stats = (
                endpoint_df
                .groupby(["endpoint", "window_id"])
                .apply(
                    lambda df: pd.concat(
                        [
                            self.poisson_window_metrics(df["total_requests"]),
                            pd.Series({"has_anomaly": df["is_anomaly"].max()})
                        ]
                    )
                )
                .reset_index()
            )

            endpoint_stats = (
                window_stats
                .groupby("endpoint")["var_mean_ratio"]
                .apply(self.poisson_endpoint_metric)
                .unstack()
                .reset_index()
            )

            endpoint_counts = (
                window_stats
                .groupby("endpoint")
                .agg(
                    n_anomaly_windows=("has_anomaly", "sum"),
                    n_windows=("has_anomaly", "count")
                )
                .reset_index()
            )

            endpoint_stats = endpoint_stats.merge(
                endpoint_counts,
                on="endpoint",
                how="left"
            )

            endpoint_stats = endpoint_stats.rename(
                columns = {
                    "var_mean_ratio_median":f"var_mean_median_{window_size}",
                    "n_anomaly_windows": f"n_anomaly_windows_{window_size}",
                    "n_windows": f"n_windows_{window_size}"
                })

            if final_score is None:
                final_score = endpoint_stats
            else:
                final_score = pd.merge(
                    final_score,
                    endpoint_stats,
                    on="endpoint",
                    how="outer"
                )

        return final_score

    def calculate_window_params(self, sample):
        window_lam = sample.mean()
        window_var = sample.var()
        return window_lam, window_var

    def plot_poisson_hist(self, ax, x, lam, sample_window_id):
        # empirical histogram
        counts = x.value_counts().sort_index()
        values = counts.index.values

        pmf = poisson.pmf(values, mu=lam) * len(x)
        ax.bar(values, counts.values, alpha=0.7, label="Empirical")
        ax.plot(values, pmf, "o-", color='orange', label=f"Poisson(λ={lam:.2f})")

        ax.set_xlabel("Requests per second")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Window {sample_window_id} – Poisson fit")
        ax.legend()

    def plot_poisson_ecdf(self, ax, x, lam, sample_window_id):
        ecdf = ECDF(x)
        k = np.arange(0, x.max() + 1)

        ax.step(k, ecdf(k), where="post", label="Empirical ECDF")
        ax.plot(k, poisson.cdf(k, mu=lam), "r--", label="Poisson CDF")

        ax.set_xlabel("Requests per second")
        ax.set_ylabel("CDF")
        ax.set_title(f"ECDF vs Poisson CDF – Window {sample_window_id}")
        ax.legend()

    def plot_q_q(self, ax, x, lam, sample_window_id):
        probs = (np.arange(1, len(x) + 1) - 0.5) / len(x)
        poisson_q = poisson.ppf(probs, mu=lam)

        ax.scatter(poisson_q, np.sort(x), s=10)
        ax.plot([0, max(poisson_q)], [0, max(poisson_q)], "r--")

        ax.set_xlabel("Poisson quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.set_title(f"Q–Q plot – Window {sample_window_id}")

    def plot_poisson_suitability_sample(self, sample_window_id, df):
        x = df[df['window_id'] == sample_window_id]['count']
        lam, var = self.calculate_window_params(x)

        print(f"λ (mean) = {lam:.2f}")
        print(f"variance = {var:.2f}")

        if lam != 0:
            print(f"var / mean = {var / lam:.2f}")

        fig = plt.figure(figsize=(10, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        self.plot_poisson_hist(ax1, x, lam, sample_window_id)
        self.plot_poisson_ecdf(ax2, x, lam, sample_window_id)
        self.plot_q_q(ax3, x, lam, sample_window_id)

        plt.tight_layout()
        plt.show()



