import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import poisson
from statsmodels.distributions.empirical_distribution import ECDF
import math


class EDA:
    def __init__(self, df, time_col='time_local', label_col='type', endpoint_col='endpoint'):
        self.df = df
        self.time_col = time_col
        self.label_col = label_col
        self.endpoint_col = endpoint_col
        self.endpoint_lookup = None
        plt.style.use('seaborn-v0_8-whitegrid')

    def plot_global_request_rate(self, frequency='1min'):
        """
        Plots global request rate evolution in time interval,
        without distinguishing endpoints
        Args:
            frequency (str): frequency of time interval
        """
        counts = (self.df.groupby([self.label_col, pd.Grouper(key=self.time_col, freq=frequency)])
                  .size().reset_index(name='count'))

        fig = px.line(counts, x=self.time_col, y='count', color=self.label_col,
                      title=f'Request frequency ({frequency})', template='plotly_white')
        fig.show()

    def set_time_axis_min(self, ax, interval):
        """
        Force a fixed interval-minute granularity on the x-axis,
        independent of the data resampling frequency.
        Args:
            ax (Axes): Axes to add the time axis to
            interval (int): interval in minutes
        """
        locator = mdates.MinuteLocator(interval=interval)
        formatter = mdates.DateFormatter("%d %H:%M")

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    def plot_endpoint_request_rate(self, valid_uris=None, top=6, frequency='min', window_size=None, interval=30, start_offset_s=0):
        """
        Generates multiple subplots for the top selected endpoints
        Args:
            valid_uris (list): list of valid endpoints
            top (int): number of top endpoints
            frequency (str): frequency of time interval to group requests
            window_size (str): window size in seconds (e.g. 30s)
            interval (int): interval in minutes to display ticks
            start_offset_s(int): total of seconds to be skipped at beginning of the interval
        """
        first_day = self.df[self.time_col].dt.normalize().min()
        df = self.df[self.df[self.time_col].dt.normalize() == first_day].copy()

        if valid_uris is None:
            valid_uris = (
                df[self.endpoint_col]
                .value_counts()
                .head(top)
                .index
            )

        df = df[df[self.endpoint_col].isin(valid_uris)].copy()
        df = df.set_index(self.time_col)

        counts = (
            df.groupby([self.endpoint_col, self.label_col])
            .resample(frequency, include_groups=False)
            .size()
            .fillna(0)
            .reset_index(name='count')
        )

        plot_start = counts['time_local'].iloc[0] + pd.to_timedelta(start_offset_s, unit='s')
        print('plot_start:', plot_start)
        counts = counts[counts[self.time_col] >= plot_start].copy()

        figsize_plot = (15, 5)
        endpoints = counts[self.endpoint_col].unique()
        n_endpoints = len(endpoints)
        n_rows = 2
        n_cols = math.ceil(n_endpoints / n_rows)
        sharex = False
        interval = interval

        if len(valid_uris) > 1:
            sharex = True
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
            df_ep = counts[counts[self.endpoint_col] == endpoint]

            for label, df_lbl in df_ep.groupby(self.label_col):
                ax.plot(
                    df_lbl[self.time_col],
                    df_lbl["count"],
                    label=label
                )

            if window_size is not None:
                t_start = df_ep[self.time_col].min().floor('s')
                t_end = df_ep[self.time_col].max()

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

            self.set_time_axis_min(ax, interval=interval)
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

    def adjust_endpoints_time_interval(self, df_valid_endpoints, t_min, t_max):
        """
        Normalizes time interval for each endpoint, creating a global time interval.
        Fills with zero the seconds that do not have requests
        Args:
            df_valid_endpoints (DataFrame): dataframe with filtered endpoints
        Returns:
            Dataframe: with global time interval, completed for each endpoint
        """
        full_time_index = pd.date_range(start=t_min, end=t_max, freq="1s", name=self.time_col)
        endpoints = df_valid_endpoints[self.endpoint_col].unique()

        full_index = pd.MultiIndex.from_product([endpoints, full_time_index], names=[self.endpoint_col, self.time_col])

        resampled = (
            df_valid_endpoints.set_index(self.time_col)
            .groupby(self.endpoint_col)
            .resample("1s")
            .agg(total_requests=("is_anomaly", "size"), is_anomaly=("is_anomaly", "max"))
            .reindex(full_index, fill_value=0)
            .reset_index()
        )
        resampled["is_anomaly"] = resampled["is_anomaly"].fillna(0).astype(int)
        return resampled

    def encode_endpoint(self, df):
        unique_eps = df['endpoint'].unique()
        mapping = {ep: f"Endpoint {i + 1}" for i, ep in enumerate(unique_eps)}
        df['endpoint'] = df['endpoint'].map(mapping)

        self.endpoint_lookup = mapping
        return df

    def _plot_poisson_hist(self, ax, x, lam, sample_window_id):
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

    def _plot_poisson_ecdf(self, ax, x, lam, sample_window_id):
        ecdf = ECDF(x)
        k = np.arange(0, x.max() + 1)

        ax.step(k, ecdf(k), where="post", label="Empirical ECDF")
        ax.plot(k, poisson.cdf(k, mu=lam), "r--", label="Poisson CDF")

        ax.set_xlabel("Requests per second")
        ax.set_ylabel("CDF")
        ax.set_title(f"ECDF vs Poisson CDF – Window {sample_window_id}")
        ax.legend()

    def _plot_q_q(self, ax, x, lam, sample_window_id):
        probs = (np.arange(1, len(x) + 1) - 0.5) / len(x)
        poisson_q = poisson.ppf(probs, mu=lam)

        ax.scatter(poisson_q, np.sort(x), s=10)
        ax.plot([0, max(poisson_q)], [0, max(poisson_q)], "r--")

        ax.set_xlabel("Poisson quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.set_title(f"Q–Q plot – Window {sample_window_id}")

    def plot_poisson_diagnostic(self, df, window_id):
        """
        Validation plot for poisson analysis: Histograms, ECDF and Q-Q Plot
        Args:
            df (DataFrame): dataset
            window_id (int): id of the analyzed window
        """
        x = df[df['window_id'] == window_id]['total_requests']
        lam = x.mean()
        var = x.var()

        print(f"λ (mean) = {lam:.2f}")
        print(f"variance = {var:.2f}")

        if lam != 0:
            print(f"var / mean = {var / lam:.2f}")

        fig = plt.figure(figsize=(10, 3))
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        self._plot_poisson_hist(ax1, x, lam, window_id)
        self._plot_poisson_ecdf(ax2, x, lam, window_id)
        self._plot_q_q(ax3, x, lam, window_id)

        plt.tight_layout()
        plt.show()

    def index_windows(self, window_indexed_df):
        window_indexed_df['window_id'] = (
                window_indexed_df['window_start']
                .astype(int)
                .rank(method='dense')
                .astype(int) - 1
        )

        window_indexed_df = window_indexed_df[['endpoint', 'window_id', 'time_local', 'total_requests', 'is_anomaly']]

        return window_indexed_df

    def poisson_window_metrics(self, sample):
        window_lam = sample.mean()
        window_var = sample.var()


        return pd.Series({
            "lambda": window_lam,
            "variance": window_var,
            "var_mean_ratio": window_var / window_lam if window_lam > 0 else np.nan,
            "n_samples": len(sample)
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
                .groupby("endpoint")
                .agg(var_mean_ratio_median=("var_mean_ratio", "median"))
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
                columns={
                    "var_mean_ratio_median": f"var_mean_median_{window_size}",
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
