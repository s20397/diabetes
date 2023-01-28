from typing import Iterable
from kedro.pipeline import Pipeline

from pipelines.data_processing_diabetes import create_pipeline

def create_diabetes_pred_pipeline() -> Iterable[Pipeline]:
    yield create_pipeline()