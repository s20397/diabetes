"""
This is a boilerplate pipeline 'data_processing_diabetes'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_diabetes, split_data, create_model, normalize_features, train_model, evaluate_model, get_score


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
                func=create_model,
                inputs="preprocessed_diabetes",
                outputs="model_created"
            ),
            node(
                func=split_data,
                inputs=["preprocessed_diabetes", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"]
            ),
            node(
                func=normalize_features,
                inputs="X_train",
                outputs="X_train_normalized"
            ),
            node(
                func=train_model,
                inputs=["model_created","X_train_normalized","y_train"],
                outputs="model_trained"
            ),
            node(
                func=normalize_features,
                inputs="X_test",
                outputs="X_test_normalized"
            ),
            node(
                func=evaluate_model,
                inputs=["model_trained","X_test_normalized","y_test"],
                outputs=None
            )
        ]
    )
