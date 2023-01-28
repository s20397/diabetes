from fastapi import FastAPI
from routers.data_processing import router

app = FastAPI(title="Diabetes", version ="0.0.1")

app.include_router(router, prefix="/data-processing")