"""
This is a boilerplate pipeline 'data_processing_diabetes'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_diabetes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_diabetes,
                inputs="diabetes",
                outputs="preprocessed_diabetes",
                name="preprocess_diabetes_node",
            ),
        ]
    )
