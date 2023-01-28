from typing import Any 

from fastapi import APIRouter, Depends
from kedro.io import DataCatalog
from kedro.pipeline import Pipeline
from kedro.runner import SequentialRunner

from dependencies.dependencies import create_diabetes_pred_pipeline
from schemas.schemas import Person, ModelEval

import pandas as pd

router = APIRouter()

@router.put('/go')
def predict_diabetes(
    persons: list[Person],
    pipeline: Pipeline = Depends(create_diabetes_pred_pipeline),
) -> ModelEval:
    runner = SequentialRunner()

    catalog = DataCatalog(
        feed_dict={"diabetes": pd.DataFrame(e.dict() for e in persons), "parameters":{"seed":101}}
    )
    result = runner.run(pipeline=pipeline, catalog=catalog)
    return ModelEval(accuracy = result["accuracy"], roc_auc = result["roc_auc"])
