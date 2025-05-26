import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get the absolute path of the directory two levels up from this file (i.e. the project root)
project_root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root_directory not in sys.path:
    sys.path.insert(0, project_root_directory)
    logger.info("Added project root (%s) to sys.path", project_root_directory)

from core.data_preparation.config import FeatureEngineeringConfig, MethodConfig
from core.data_preparation.pipeline import DataPrepPipelineBuilder
import pandas as pd

def main():
    logger.info("Creating configuration for the pipeline.")
    config = FeatureEngineeringConfig(
        data_preparation_steps=[
            #MethodConfig(name="sklearn.preprocessing.Normalizer", params=dict(norm="l2")),
            MethodConfig(
                name="core.data_preparation.pipeline.DataVolumePreparation",
                params=dict(
                    raw_csv_path="../data/raw/data_call_center.csv",
                    #output_path="data/processed",
                ),
            )
        ],
    )

    logger.info("Building the pipeline.")
    pipeline_builder = DataPrepPipelineBuilder(config)
    pipeline = pipeline_builder.get_pipeline()

    logger.info("Transforming data with the pipeline.")
    X_transformed = pipeline.transform(None)

    logger.info("Transformed data columns: %s", X_transformed.columns)
    print(X_transformed.columns)

if __name__ == '__main__':
    main()