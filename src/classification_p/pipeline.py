"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import save_model, train_model, preprocess_data, load_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_data,
                inputs=["external_sources_file_path", "internal_data_file_path"],
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
                inputs=["rf_model"],
                outputs=None,
                name="save_model_node",
            ),
        ]
    )
