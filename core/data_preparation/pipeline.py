import sys
import os
import json
import pandas as pd
from glob import glob
import concurrent.futures
import importlib
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from core.data_preparation.config import FeatureEngineeringConfig, MethodConfig

import numpy as np
import holidays
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPrepPipelineBuilder:
    """
    Builds a data preparation pipeline based on a given feature engineering configuration.

    Attributes:
        _config (FeatureEngineeringConfig): The feature engineering configuration.
    """

    def __init__(self, config_file: FeatureEngineeringConfig):
        """
        Initializes the pipeline builder with the specified configuration.

        Args:
            config_file (FeatureEngineeringConfig): Configuration for feature engineering.
        """
        self._config = config_file

    def _get_method(self, step: MethodConfig):
        """
        Instantiates a transformation method defined in the configuration.

        Parses the module and class name from the provided configuration, imports the module,
        retrieves the class, and initializes it with the given parameters.

        Args:
            step (MethodConfig): The configuration for the method.

        Returns:
            object: An instance of the method class.
        """
        module_name, class_name = step.name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        method_class = getattr(module, class_name)
        return method_class(**step.params)

    def _build_pipeline(self) -> Pipeline:
        """
        Constructs a scikit-learn Pipeline based on the configuration's list of data preparation steps.

        Returns:
            Pipeline: A pipeline object consisting of all configured transformation steps.
        """
        steps = []
        for step in self._config.data_preparation_steps:
            method_instance = self._get_method(
                MethodConfig(name=step.name, params=step.params)
            )
            # Use the class name as the step name in the pipeline.
            steps.append((method_instance.__class__.__name__, method_instance))
        return Pipeline(steps)

    def get_pipeline(self) -> Pipeline:
        """
        Retrieves the complete data preparation pipeline as configured.

        Returns:
            Pipeline: A scikit-learn Pipeline that applies the specified transformations in order.
        """
        return self._build_pipeline()


class DropColumns(BaseEstimator, TransformerMixin):
    """
    Transformer that drops specified columns from a pandas DataFrame.

    Attributes:
        columns (list): A list of column names to be removed from the DataFrame.
    """

    def __init__(self, columns: list = None):
        """
        Initializes the transformer with a list of columns to drop.

        Args:
            columns (list, optional): Columns to drop. Defaults to None.
        """
        self.columns = columns

    def transform(self, X):
        """
        Drops the configured columns from the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with the specified columns removed.

        Raises:
            TypeError: If X is not a DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            logger.error("X must be a DataFrame.")
            raise TypeError("X must be a DataFrame.")
        for c in self.columns:
            if c in X.columns:
                X = X.drop(c, axis=1)
        return X


class DeepPreprocess:

    def transform(self, X) -> pd.DataFrame:

        X["anomalyscore"] = X["anomalyscore"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
        )

        return X

    def fit(self, X):
        """
        Dummy fit method to comply with the scikit-learn transformer interface.

        Args:
            X: Input data (unused).

        Returns:
            DeepPreprocess: The fitted processor instance.
        """
        return self


class DataVolumePreparation(BaseEstimator, TransformerMixin):
    """
    Transformer that prepares raw call center data for modeling by performing the following steps:
      - Cleaning and preprocessing the raw data,
      - Processing closure reason values and grouping them into categories,
      - Aggregating data on a daily basis (including computing a 30-day rolling call count target),
      - Adding a binary feature indicating Brazilian holidays.

    The transformer accepts a pandas DataFrame directly; if none is provided, it loads the data from a CSV file
    specified by the raw_csv_path parameter.

    Attributes:
        raw_csv_path (str): Optional path to the CSV file containing raw call center data.
        train_split_date (str): Date used for splitting the data, if necessary.
    """

    def __init__(self, raw_csv_path: str = None, train_split_date: str = "2025-03-01"):
        """
        Args:
            raw_csv_path (str, optional): Path to the raw CSV file.
                If provided and transform is called with None, the CSV is loaded.
            train_split_date (str): Date string to split train and test if needed.
        """
        self.raw_csv_path = raw_csv_path
        self.train_split_date = pd.Timestamp(train_split_date)

    def fit(self, X, y=None):
        # No fitting needed for this transformer.
        return self

    def transform(self, X=None):
        """
        Executes the full data preparation pipeline.

        If X is None and self.raw_csv_path is provided, the raw data is loaded.
        Otherwise, X should be a pandas DataFrame.

        Returns:
            pd.DataFrame: The fully transformed DataFrame with daily features.
        """
        # Load raw data if necessary.
        if X is None:
            if self.raw_csv_path is None:
                raise ValueError("No input DataFrame or raw_csv_path provided.")
            X = pd.read_csv(self.raw_csv_path)

        # Step 1: Clean the data.
        df = self._clean_data(X)

        # Step 2: Process closure_reason values.
        df = self._process_closure_reason(df)

        # Step 3: Aggregate by day and calculate target_next_30days.
        daily_df = self._adjust_target_variable(df)

        # Step 4: Add Brazilian holiday feature.
        daily_df = self._create_brazil_holiday_feature(daily_df, date_col="date")

        # Reset index to make 'date' a column.
        daily_df.reset_index(inplace=True)
        return daily_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop unneeded identifier columns.
        df_processed = df.drop(
            columns=[
                "contract_number",
                "occurrence_number",
                "closure_login",
                "occurrence_type_ID",
                "internal_sla_end_date",
            ],
            errors="ignore",
        )
        # Convert date columns.
        df_processed["occurrence_date"] = pd.to_datetime(
            df_processed["occurrence_date"],
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce",
        )
        df_processed["closure_date"] = pd.to_datetime(
            df_processed["closure_date"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
        )
        # Fill missing closure_date with current date + 30 days.
        df_processed.loc[df_processed["closure_date"].isnull(), "closure_date"] = (
            pd.Timestamp.today() + pd.Timedelta(days=30)
        )
        # Create time_to_resolve (in hours)
        df_processed["time_to_resolve"] = (
            df_processed["closure_date"] - df_processed["occurrence_date"]
        ).dt.total_seconds() / 3600
        # Fill missing closure_reason
        df_processed.loc[df_processed["closure_reason"].isnull(), "closure_reason"] = (
            "Unknown"
        )
        return df_processed

    def _process_closure_reason(self, df: pd.DataFrame) -> pd.DataFrame:
        # Group similar reasons under common labels.
        def group_reason(reason):
            value = str(reason).upper().strip()
            if value == "DESISTENCIA":
                return "desistencia"
            if "TIMEOUT" in value:
                return "timeout"
            if "ME NAO GERADA" in value:
                return "me_n_gerada"
            if "ME GERADA" in value:
                return "me_gerada"
            if "NAO LIBERADO" in value:
                return "n_liberado"
            if "LIBERADO" in value:
                return "liberado"
            if "ABERTURA" in value:
                return "abertura"
            if "DIRECIONADO" in value:
                return "direcionado"
            if "MUD" in value:
                return "mudanca"
            if "ERRO" in value:
                return "erro"
            if "OCORRENCIA" in value:
                return "ocorrencia"
            if "UNKNOWN" in value:
                return "unknown"
            return value

        df["aux"] = df["closure_reason"].apply(group_reason)
        freq = df["aux"].value_counts(normalize=True)
        df["grupo_reason_final"] = df["aux"].apply(
            lambda x: x if freq[x] >= 0.001140 else "Outros"
        )
        df["grupo_reason_final"] = df["grupo_reason_final"].apply(
            lambda x: unicodedata.normalize("NFKD", x)
            .encode("ASCII", "ignore")
            .decode("utf-8")
            .lower()
            .strip()
            .replace(" ", "_")
        )
        df = pd.get_dummies(df, columns=["grupo_reason_final"], prefix="closure")
        df.drop(
            columns=["aux", "closure_date", "closure_reason"],
            inplace=True,
            errors="ignore",
        )
        return df

    def _adjust_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups data by day (using occurrence_date) to calculate aggregated metrics and
        computes target_next_30days as the sum of call_count over the next 30 days.
        """
        agg_dict = {
            "time_to_resolve": "mean",
            "closure_aberta_ca1-end_(tempo_de_contato_excedido)": "sum",
            "closure_abertura": "sum",
            "closure_desistencia": "sum",
            "closure_direcionado": "sum",
            "closure_liberado": "sum",
            "closure_mdu_encaminhado_para_vistoria": "sum",
            "closure_me_com_alteracao_de_pacote": "sum",
            "closure_me_gerada": "sum",
            "closure_me_n_gerada": "sum",
            "closure_n_liberado": "sum",
            "closure_outros": "sum",
            "closure_queda_de_ligacao": "sum",
            "closure_reencaminhado_para_vistoria": "sum",
            "closure_timeout": "sum",
            "closure_unknown": "sum",
        }
        # Group by day using occurrence_date.
        daily_agg = df.groupby(pd.Grouper(key="occurrence_date", freq="D")).agg(
            agg_dict
        )
        # Calculate daily call_count.
        daily_count = (
            df.groupby(pd.Grouper(key="occurrence_date", freq="D"))
            .size()
            .rename("call_count")
        )
        daily_df = daily_agg.join(daily_count)
        # Reindex so every day in range appears.
        all_dates = pd.date_range(
            start=daily_df.index.min(), end=daily_df.index.max(), freq="D"
        )
        daily_df = daily_df.reindex(all_dates, fill_value=0)
        daily_df.index.name = "date"
        # Calculate target as sum of call_count over next 30 days (including current day).
        daily_df["target_next_30days"] = (
            daily_df.iloc[::-1]["call_count"]
            .rolling(window="30D", min_periods=1)
            .sum()
            .iloc[::-1]
        )
        # Add year and month columns.
        daily_df["year"] = daily_df.index.year
        daily_df["month"] = daily_df.index.month
        return daily_df

    def _create_brazil_holiday_feature(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Adds a binary column 'is_brazil_holiday' to denote if a given date is a Brazilian holiday.
        """

        df.reset_index(inplace=True)
        df[date_col] = pd.to_datetime(df[date_col])
        years = df[date_col].dt.year.unique().tolist()
        br_holidays = holidays.Brazil(years=years)
        df["is_brazil_holiday"] = df[date_col].apply(
            lambda d: 1 if d in br_holidays else 0
        )
        return df
