"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import save_model, train_model, preprocess_data, load_data, init_spark


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=init_spark,
                inputs=None,
                outputs="spark",
                name="init_spark",
            ),
            node(
                func=load_data,
                inputs=["spark","external_sources_file_path", "internal_data_file_path"],
                outputs="df_full",
                name="load_data_node",
            ),
            node(
                func=preprocess_data,
                inputs=["df_full"],
                outputs="df_cleaned",
                name="preprocess_data_node",
            ),
            node(
                func=train_model,
                inputs=["df_cleaned"],
                outputs=["rf_model", "accuracy"],
                name="train_model_node",
            ),
            node(
                func=save_model,
                inputs=["rf_model", "save_output_parquet_path"],
                outputs=None,
                name="save_model_node",
            ),
        ]
    )
