"""
FastAPI main application entry point for the Ceiling ML Pipeline.

REST endpoints for:
- Running the ML pipeline
- Pipeline status and metrics
- Health checks
"""
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time
import uuid

from src.models.data_models import (
    CeilingGrid, ComponentType, PipelineResult
)
from src.pipeline.orchestrator import (
    CeilingMLPipeline, create_default_pipeline,
    PipelineConfig, StageConfig, ModelConfig, SelectionStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[CeilingMLPipeline] = None

# Job tracking for async execution
jobs: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipeline
    logger.info("Initializing ML pipeline...")
    pipeline = create_default_pipeline()
    pipeline.initialize()
    logger.info("ML pipeline ready")
    yield
    logger.info("Shutting down ML pipeline...")


app = FastAPI(
    title="Ceiling ML Pipeline API",
    description="API for running ML-based ceiling component placement",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class GridInput(BaseModel):
    """Input model for ceiling grid."""
    room_id: Optional[str] = None
    width: int = Field(..., ge=1, le=100, description="Grid width in cells")
    height: int = Field(..., ge=1, le=100, description="Grid height in cells")
    matrix: List[List[int]] = Field(
        ...,
        description="2D matrix: 0=valid, 1=invalid, 2=light, 3=air_supply, 4=air_return, 5=smoke_detector"
    )
    cell_size: float = Field(default=2.0, description="Cell size in feet")
    
    class Config:
        json_schema_extra = {
            "example": {
                "room_id": "room-123",
                "width": 10,
                "height": 8,
                "matrix": [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                "cell_size": 2.0
            }
        }


class PipelineResponse(BaseModel):
    """Response model for pipeline execution."""
    room_id: str
    input_matrix: List[List[int]]
    output_matrix: List[List[int]]
    execution_time_ms: float
    stages_executed: int
    stage_summary: Dict[str, Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "room_id": "room-123",
                "input_matrix": [[0, 0], [0, 0]],
                "output_matrix": [[2, 0], [0, 5]],
                "execution_time_ms": 45.2,
                "stages_executed": 4,
                "stage_summary": {
                    "light_placement": {
                        "model_selected": "light_placement_v2",
                        "placements": 4,
                        "confidence": 0.92
                    }
                }
            }
        }


class JobResponse(BaseModel):
    """Response for async job submission."""
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Status of an async job."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float
    completed_at: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pipeline_initialized: bool
    pipeline_version: Optional[str]
    models_loaded: int


class StageConfigInput(BaseModel):
    """Input for configuring a pipeline stage."""
    stage_name: str
    component_type: str
    model_ids: List[str]
    selection_strategy: str = "highest_confidence"
    parallel_execution: bool = True


class PipelineConfigInput(BaseModel):
    """Input for custom pipeline configuration."""
    stages: List[StageConfigInput]


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and pipeline health status."""
    return HealthResponse(
        status="healthy" if pipeline and pipeline._is_initialized else "degraded",
        pipeline_initialized=pipeline._is_initialized if pipeline else False,
        pipeline_version=pipeline.config.pipeline_version if pipeline else None,
        models_loaded=sum(len(s.models) for s in pipeline.stages) if pipeline else 0
    )


@app.post("/pipeline/run", response_model=PipelineResponse, tags=["Pipeline"])
async def run_pipeline(input_data: GridInput):
    """
    Run the ML pipeline synchronously on the input grid.
    
    This endpoint executes all pipeline stages and returns the
    completed building JSON with all components placed.
    """
    if not pipeline or not pipeline._is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Convert input to CeilingGrid
        grid = CeilingGrid.from_matrix(
            input_data.matrix,
            room_id=input_data.room_id
        )
        
        # Execute pipeline
        result = pipeline.execute(grid)
        
        # Build stage summary
        stage_summary = {}
        for stage_name, selected in result.selected_results.items():
            stage_summary[stage_name] = {
                "model_selected": selected.model_id,
                "model_version": selected.model_version,
                "placements": len(selected.placements),
                "confidence": selected.confidence_score,
                "execution_time_ms": selected.execution_time_ms,
                "is_valid": selected.is_valid
            }
        
        return PipelineResponse(
            room_id=result.room_id,
            input_matrix=result.input_grid.to_matrix(),
            output_matrix=result.output_grid.to_matrix(),
            execution_time_ms=result.total_execution_time_ms,
            stages_executed=len(result.selected_results),
            stage_summary=stage_summary
        )
        
    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline/run/async", response_model=JobResponse, tags=["Pipeline"])
async def run_pipeline_async(
    input_data: GridInput,
    background_tasks: BackgroundTasks
):
    """
    Submit a pipeline job for asynchronous execution.
    
    Returns a job ID that can be used to check status and retrieve results.
    Useful for large grids or when non-blocking execution is needed.
    """
    job_id = str(uuid.uuid4())
    
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": time.time(),
        "completed_at": None
    }
    
    def execute_job():
        jobs[job_id]["status"] = "running"
        try:
            grid = CeilingGrid.from_matrix(
                input_data.matrix,
                room_id=input_data.room_id
            )
            result = pipeline.execute(grid)
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = {
                "room_id": result.room_id,
                "output_matrix": result.output_grid.to_matrix(),
                "execution_time_ms": result.total_execution_time_ms
            }
            jobs[job_id]["completed_at"] = time.time()
            
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["completed_at"] = time.time()
    
    background_tasks.add_task(execute_job)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Job submitted successfully"
    )


@app.get("/pipeline/job/{job_id}", response_model=JobStatus, tags=["Pipeline"])
async def get_job_status(job_id: str):
    """Get the status of an async pipeline job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result=job["result"],
        error=job["error"],
        created_at=job["created_at"],
        completed_at=job["completed_at"]
    )


@app.get("/pipeline/stages", tags=["Pipeline"])
async def get_pipeline_stages():
    """Get information about configured pipeline stages."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    stages = []
    for stage in pipeline.stages:
        stages.append({
            "stage_name": stage.config.stage_name,
            "component_type": stage.config.component_type.value,
            "models": [m.config.model_id for m in stage.models],
            "selection_strategy": stage.config.selection_strategy.value,
            "parallel_execution": stage.config.parallel_execution
        })
    
    return {
        "pipeline_id": pipeline.config.pipeline_id,
        "pipeline_version": pipeline.config.pipeline_version,
        "stages": stages
    }


@app.get("/models", tags=["Models"])
async def list_models():
    """List all available models in the registry."""
    from src.models.placement_models import MODEL_REGISTRY
    
    return {
        "available_models": list(MODEL_REGISTRY.keys())
    }


# ============================================================================
# Metrics endpoint (for Prometheus scraping)
# ============================================================================

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Expose metrics in Prometheus format.
    
    In production, you would use prometheus_client library for proper formatting.
    This is a simplified version for demonstration.
    """
    metrics = []
    
    # Pipeline health
    metrics.append(f'pipeline_initialized{{pipeline_id="{pipeline.config.pipeline_id}"}} {1 if pipeline._is_initialized else 0}')
    
    # Models per stage
    for stage in pipeline.stages:
        metrics.append(
            f'stage_models_count{{stage="{stage.config.stage_name}"}} {len(stage.models)}'
        )
    
    # Job counts
    job_counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
    for job in jobs.values():
        job_counts[job["status"]] += 1
    
    for status, count in job_counts.items():
        metrics.append(f'jobs_total{{status="{status}"}} {count}')
    
    return "\n".join(metrics)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)