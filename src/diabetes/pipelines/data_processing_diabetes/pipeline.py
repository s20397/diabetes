"""
This is a boilerplate pipeline 'data_processing_diabetes'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_diabetes, splitData, createModel, normalizeFeatures, trainModel, evaluateModel


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_diabetes,
                inputs="diabetes",
                outputs="preprocessed_diabetes",
                name="preprocess_diabetes_node",
            ),
            node(
                func=splitData,
                inputs="preprocessed_diabetes",
                outputs=["X_train", "X_test", "y_train", "y_test"]
            ),
            node(
                func=createModel,
                inputs=[],
                outputs="model"
            ),
            node(
                func=normalizeFeatures,
                inputs="X_train",
                outputs="X_train_normalized"
            ),
            node(
                func=trainModel,
                inputs=["model","X_train_normalized","y_train"],
                outputs="trainedModel"
            ),
            node(
                func=normalizeFeatures,
                inputs="X_test",
                outputs="X_test_normalized"
            ),
            node(
                func=evaluateModel,
                inputs=["trainedModel","X_test_normalized","y_test"],
                outputs=None
            )
        ]
    )
