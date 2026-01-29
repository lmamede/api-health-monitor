import pandas as pd
import os

class DataManager:
    def __init__(self):
        self.timestamp_col = None
        self.label_col = None
        self.type_col = None
        self.dst_port_col = None
        self.dst_ip_col = None
        self.src_ip_col = None

    def get_all_datasets_path(self):
        """ Autmatically obtain the datasets paths """
        csv_paths = []
        files_codes = ['8', '9', '10', '11']  # datasets used
        with os.scandir('../datasets/Processed_Network_dataset') as ton_dts:
            for ton_dt in ton_dts:
                csv_paths = csv_paths + [ton_dt.path for code in files_codes if ton_dt.name.endswith(f'{code}.csv')]
        return csv_paths

    def merge_datasets(self, paths):
        """ Merges all the datasets into a single dataframe """
        dfs = []
        for path in paths:
            df_i = pd.read_csv(path, low_memory=False)
            dfs.append(df_i)
        df = pd.concat(dfs, ignore_index=True)
        return df

    def load_data(self):
        csv_paths = self.get_all_datasets_path()
        return self.merge_datasets(csv_paths)

    def format_columns(self, df):
        """Converting columns to represent HTTP endpoints and formating data"""
        df = df[[self.src_ip_col,
                 self.dst_ip_col,
                 self.dst_port_col,
                 self.timestamp_col,
                 self.type_col,
                 self.label_col]].copy()

        # parse timestamp
        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], unit='s', errors='coerce')
        df = df.sort_values(self.timestamp_col)
        df = df.dropna(subset=[self.timestamp_col])

        # create normalized time
        df[self.timestamp_col] = df[self.timestamp_col].dt.floor('s')

        # creating endpoint column
        df['endpoint'] = df[self.dst_ip_col] + ':' + df[self.dst_port_col].astype(str)
        df = df[[self.src_ip_col, 'endpoint', self.timestamp_col, self.type_col, self.label_col]]  # original names
        df.columns = ['address', 'endpoint', 'time_local', 'type', 'is_anomaly']  # converted names

        # format columns
        df['address'] = df['address'].astype(str)
        df['endpoint'] = df['endpoint'].astype(str)
        df['type'] = df['type'].astype(str)
        df['is_anomaly'] = df['is_anomaly'].astype(int)

        return df

    def select_features(self, src_ip_col, dst_ip_col, dst_port_col, 
                        type_col, label_col,timestamp_col):
        self.src_ip_col = src_ip_col
        self.dst_ip_col = dst_ip_col
        self.dst_port_col = dst_port_col
        self.type_col = type_col
        self.label_col = label_col
        self.timestamp_col = timestamp_col
        