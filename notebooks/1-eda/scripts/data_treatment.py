import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class DataManager:
    """
    Manages data lifecycle: load, fusion,and standardization of
    network traffic datasets
    """
    def __init__(self, data_dir):
        self.column_map = {}
        self.output_cols = [
            'address',
            'endpoint',
            'time_local',
            'type',
            'is_anomaly'
        ]
        self.data_dir=data_dir
        self._dataset=None

    def select_features(self, **kwargs):
        """
        Defines original columns mapping
        E.g.: select_features(src_ip_col='Src IP', dst_ip_col='Dst IP', ...)
        """
        self.column_map = kwargs

    def _get_all_datasets_path(self, files_ends, dataset_dir):
        """
        Search for csv files path
        Args:
            files_ends (list): files to be searched
            dataset_dir (str): dataset dir
        Returns:
            csv_paths (list): csv files path
        """
        if not os.path.exists(dataset_dir):
            logging.error(f"Path not found: {dataset_dir}")
            return []

        csv_paths = [
            entry.path for entry in os.scandir(dataset_dir)
            if entry.is_file() and any(entry.name.endswith(f'{ext}.csv') for ext in files_ends)
        ]

        logging.info(f"Found files ({len(csv_paths)}): {csv_paths}")
        return csv_paths

    def _merge_datasets(self, paths):
        """
        Reads and concatenates multiple csv files
        Args:
            paths (list): csv files path to be merged
        Returns:
            Dataframe: merged dataframe
        """
        if not paths:
            return pd.DataFrame()

        dfs = []
        for path in paths:
            try:
                df_i = pd.read_csv(path, low_memory=False)
                dfs.append(df_i)
            except Exception as e:
                logging.warning(f"Fail to read {path}: {e}")

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def get_dataset(self, files_ends):
        """
        Searches and load datasets
        Args:
            files_ends (list): files to be searched
        Returns:
            Dataframe: merged dataframe
        """
        if self._dataset is None:
            paths = self._get_all_datasets_path(files_ends,self.data_dir)
            self._dataset = self._merge_datasets(paths)
        return self._dataset

    def _format_timestamp(self, df):
        """
        Converts timestamp to datetime, removes missing values, and
        normalize to seconds
        Args:
            df (DataFrame): dataframe to be formatted
        Returns:
            DataFrame: formatted dataframe
        """
        ts_col = self.column_map['timestamp_col']
        df[ts_col] = pd.to_datetime(df[ts_col], unit='s', errors='coerce')
        df = df.dropna(subset=[ts_col]).sort_values(ts_col)
        df[ts_col] = df[ts_col].dt.floor('s')
        return df

    def _feature_engineering(self, df):
        """
        Creates new column 'endpoint' from ip and port columns
        endpoint = IP_Destination:Port_Destination
        Args:
            df (DataFrame): dataframe to be formatted
        Returns:
            DataFrame: with new feature added
        """
        dst_ip = self.column_map['dst_ip_col']
        dst_port = self.column_map['dst_port_col']
        df['endpoint'] = df[dst_ip].astype(str) + ':' + df[dst_port].astype(str)
        return df

    def _feature_renaming(self, df):
        """
        Renames column to be consistent with their meaning
        Args:
            df (DataFrame): dataframe to be renamed
        Returns:
            DataFrame: with new features names
        """
        rename_dict = {
            self.column_map['src_ip_col']: 'address',
            self.column_map['timestamp_col']: 'time_local',
            self.column_map['type_col']: 'type',
            self.column_map['label_col']: 'is_anomaly'
        }

        df = df.rename(columns=rename_dict)
        return df[self.output_cols]

    def _cast_features(self, df):
        """
        Assigns the correct type for each feature
        Args:
            df (DataFrame): dataframe to be treated
        Returns:
            DataFrame: formatted dataframe
        """
        return df.assign(
            address=df['address'].astype(str),
            endpoint=df['endpoint'].astype(str),
            type=df['type'].astype(str),
            is_anomaly=df['is_anomaly'].fillna(0).astype(int)
        )

    def transform_features(self, df):
        """
        Data transformation: cleaning, format, and feature creation
        Args:
            df (pd.DataFrame): dataframe to be formatted
        Returns:
            Dataframe: formatted dataframe
        """
        if df.empty:
            return df

        cols_to_extract = [
            self.column_map['src_ip_col'],
            self.column_map['dst_ip_col'],
            self.column_map['dst_port_col'],
            self.column_map['timestamp_col'],
            self.column_map['type_col'],
            self.column_map['label_col']
        ]

        df = df[cols_to_extract].copy()

        df = self._format_timestamp(df)
        df = self._feature_engineering(df)
        df = self._feature_renaming(df)
        df = self._cast_features(df)

        return df