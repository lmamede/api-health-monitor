import pandas as pd

class TrafficLearner:
    """
    Learns traffic normal behavior
    """
    def __init__(self, window_sizes, path_normal_traffic_df):
        self.window_sizes = window_sizes
        self.path_normal_traffic_df = path_normal_traffic_df
        self.raw_normal_traffic_df = None

    def get_normal_traffic_df(self):
        """
        Loads, format timestamp and creates a cache for
        normal traffic dataset
        """
        if self.raw_normal_traffic_df is None:
            self.raw_normal_traffic_df = pd.read_csv(self.path_normal_traffic_df, low_memory=False)
            self.raw_normal_traffic_df['time_local'] = pd.to_datetime(self.raw_normal_traffic_df['time_local'])
        return self.raw_normal_traffic_df

    def index_windows(self, df):
        """
        Computes window size to index request to each
        corresponding window

        Args:
            df (DataFrame): traffic request frequency by second

        Returns:
            dataframe with window indexed dataset
        """
        windows_df = []

        for window_size in self.window_sizes:
            df['window_start'] = df['time_local'].dt.floor(window_size).copy()
            df['window_id'] = (
                    df['window_start']
                    .astype(int)
                    .rank(method='dense')
                    .astype(int) - 1
            )

            df['window_size'] = int(window_size.replace('s', ''))

            windows_df.append(
                df[[
                    'endpoint',
                    'window_size',
                    'window_id',
                    'window_start',
                    'time_local',
                    'total_requests',
                    'is_anomaly'
                ]]
            )

        return pd.concat(windows_df, ignore_index=True)

    def learn_traffic_information(self):
        """
        Estimates lambda using each endpoint
        window mean request frequency

        Returns:
            DataFrame: with computed lambda per window
        """
        df = self.get_normal_traffic_df()
        df = self.index_windows(df)

        window_stats = (
            df.groupby(["window_size", "endpoint", "window_id"])
            .agg(lam=("total_requests", "mean"))
            .reset_index()
        )

        return window_stats

    def label_test_windows(self, df):
        df = self.index_windows(df)

        window_stats = (
            df.groupby(["window_size","endpoint", "window_id"])
            .agg(has_anomaly=("is_anomaly", "max"))
            .reset_index()
        )

        return window_stats
