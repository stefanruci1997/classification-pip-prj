"""
This is a boilerplate pipeline 'classification_pipeline'
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import save_model, train_model, preprocess_data, load_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_data,
                ["params:external_sources_file_path", "params:internal_data_file_path"],
                "df_full",
                name="load_data_node",
            ),
            node(
                preprocess_data,
                ["df_full"],
                "df_cleaned",
                name="preprocess_data_node",
            ),
            node(
                train_model,
                ["df_cleaned"],
                ["rf_model", "accuracy"],
                name="train_model_node",
            ),
            node(
                func=save_model,
                inputs=["rf_model", "params:save_output_parquet_path"],
                outputs= None,
                name="save_model_node",
            ),
        ]

    )
