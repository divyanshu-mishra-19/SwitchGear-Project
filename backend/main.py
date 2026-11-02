"""
CircuitGuard AI - FastAPI Backend Application
Exposes REST endpoints for model health, metadata, feature schemas,
demonstration scenarios, single/batch inference, and multi-model comparison.
Also serves the React-based frontend dashboard.
"""

import os
import sys

# Prevent OpenMP collision on macOS ARM64
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import io
import json
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.schemas import (
    CircuitBreakerInput,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from backend.inference import get_engine

# Lifespan event to load model artifacts once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI starting up: loading ML inference engine...")
    get_engine()
    print("ML inference engine loaded successfully!")
    yield
    print("FastAPI shutting down.")

app = FastAPI(
    title="CircuitGuard AI API",
    description="Predictive Maintenance API for High-Voltage Substation Circuit Breakers",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend and local integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    engine = get_engine()
    available = engine.get_available_models()
    return HealthResponse(
        status="healthy",
        models_loaded=len(available) > 0,
        available_models=available,
        device="macOS M4 (Accelerate / MPS / CPU)",
        version="1.0.0"
    )

@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
def get_model_info():
    engine = get_engine()
    return ModelInfoResponse(
        project_name="CircuitGuard AI",
        primary_model="TabNet (Transfer Learning)",
        transfer_learning_approach="Self-supervised feature reconstruction pretraining followed by supervised classification fine-tuning",
        target_classes={
            "0": "No Maintenance Required",
            "1": "Maintenance Required"
        },
        input_features_count=len(engine.metadata["raw_features_order"]),
        test_set_evaluation=engine.metrics,
        available_models=engine.get_available_models()
    )

@router.get("/features", tags=["Features"])
def get_features_schema():
    engine = get_engine()
    return {
        "raw_features_order": engine.metadata["raw_features_order"],
        "numerical_features": engine.metadata["numerical_features"],
        "categorical_features": engine.metadata["categorical_features"],
        "target_col": engine.metadata["target_col"],
        "target_classes": engine.metadata["target_classes"]
    }

@router.get("/scenarios", tags=["Demonstration Scenarios"])
def get_demonstration_scenarios():
    scenarios_path = os.path.join("models", "scenarios.json")
    if not os.path.exists(scenarios_path):
        raise HTTPException(status_code=404, detail="Demonstration scenarios not found")
    with open(scenarios_path, "r") as f:
        scenarios = json.load(f)
    return scenarios

@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    engine = get_engine()
    try:
        data_dict = request.features.model_dump()
        result = engine.predict_single(data_dict, model_name=request.model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@router.post("/compare-models", tags=["Inference"])
def compare_models(features: CircuitBreakerInput):
    engine = get_engine()
    try:
        data_dict = features.model_dump()
        result = engine.compare_all_models(data_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model comparison error: {str(e)}")

@router.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    file: Optional[UploadFile] = File(None),
    model_name: Optional[str] = Query("TabNet (Transfer Learning)")
):
    engine = get_engine()
    if file is None:
        raise HTTPException(status_code=400, detail="Please upload a CSV file with circuit breaker readings")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty")

        result = engine.predict_batch(df, model_name=model_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch processing error: {str(e)}")

# Mount routes under both root and /api
app.include_router(router)
app.include_router(router, prefix="/api")

# Serve React Dashboard from frontend/dist if built
dist_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.head("/", include_in_schema=False)
    async def serve_root():
        return FileResponse(os.path.join(dist_dir, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_react_app(full_path: str):
        # Allow API docs and OpenAPI schema through
        if full_path.startswith("docs") or full_path.startswith("openapi.json"):
            raise HTTPException(status_code=404)
        file_path = os.path.join(dist_dir, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(dist_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
