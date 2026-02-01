"""
Pipeline management routes.
"""

from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel

from kggen_cuad.config import get_settings
from kggen_cuad.models.api import PipelineStatusResponse
from kggen_cuad.pipeline.orchestrator import (
    PipelineOrchestrator,
    PipelineStatus,
    get_pipeline_orchestrator,
)
from kggen_cuad.storage.redis_cache import get_redis_cache

logger = structlog.get_logger(__name__)
router = APIRouter()


class PipelineRunRequest(BaseModel):
    """Pipeline run request."""
    contract_paths: list[str]
    skip_resolution: bool = False
    persist_graph: bool = True


class PipelineTextRequest(BaseModel):
    """Pipeline text processing request."""
    text: str
    cuad_id: str
    persist: bool = True


@router.post("/run")
async def run_pipeline(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """
    Start a pipeline run for multiple contracts.

    Returns immediately with pipeline ID for status tracking.
    """
    orchestrator = get_pipeline_orchestrator()
    cache = get_redis_cache()

    # Convert paths
    paths = [Path(p) for p in request.contract_paths]

    # Validate paths exist
    for path in paths:
        if not path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Contract file not found: {path}",
            )

    # Generate pipeline ID
    from uuid import uuid4
    pipeline_id = uuid4()

    # Set initial status
    cache.set_contract_status(str(pipeline_id), PipelineStatus.PENDING.value)

    # Run in background
    async def run_background():
        try:
            cache.set_contract_status(str(pipeline_id), PipelineStatus.EXTRACTING.value)

            result = await orchestrator.run(
                contract_paths=paths,
                skip_resolution=request.skip_resolution,
                persist_graph=request.persist_graph,
            )

            cache.set_contract_status(str(pipeline_id), result.status.value)

            # Store result summary
            cache.set(
                f"pipeline:{pipeline_id}:result",
                result.to_dict(),
                ttl=3600,
            )

        except Exception as e:
            logger.error("pipeline_background_failed", error=str(e))
            cache.set_contract_status(str(pipeline_id), PipelineStatus.FAILED.value)

    background_tasks.add_task(run_background)

    return {
        "pipeline_id": str(pipeline_id),
        "status": PipelineStatus.PENDING.value,
        "contracts": len(paths),
        "message": "Pipeline started. Use /pipeline/status/{pipeline_id} to track progress.",
    }


@router.post("/upload-and-run")
async def upload_and_run_pipeline(
    files: list[UploadFile] = File(...),
    skip_resolution: bool = False,
    persist_graph: bool = True,
    background_tasks: BackgroundTasks = None,
) -> dict[str, Any]:
    """
    Upload contracts and run pipeline.
    """
    settings = get_settings()
    orchestrator = get_pipeline_orchestrator()
    cache = get_redis_cache()

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Save uploaded files
    temp_paths = []
    try:
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files accepted: {file.filename}",
                )

            temp_path = Path(settings.temp_dir) / file.filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)

            content = await file.read()
            temp_path.write_bytes(content)
            temp_paths.append(temp_path)

        # Generate pipeline ID
        from uuid import uuid4
        pipeline_id = uuid4()

        cache.set_contract_status(str(pipeline_id), PipelineStatus.PENDING.value)

        # Run in background
        async def run_background():
            try:
                cache.set_contract_status(str(pipeline_id), PipelineStatus.EXTRACTING.value)

                result = await orchestrator.run(
                    contract_paths=temp_paths,
                    skip_resolution=skip_resolution,
                    persist_graph=persist_graph,
                )

                cache.set_contract_status(str(pipeline_id), result.status.value)
                cache.set(f"pipeline:{pipeline_id}:result", result.to_dict(), ttl=3600)

            except Exception as e:
                logger.error("pipeline_upload_run_failed", error=str(e))
                cache.set_contract_status(str(pipeline_id), PipelineStatus.FAILED.value)

            finally:
                # Cleanup temp files
                for path in temp_paths:
                    if path.exists():
                        path.unlink()

        if background_tasks:
            background_tasks.add_task(run_background)
        else:
            await run_background()

        return {
            "pipeline_id": str(pipeline_id),
            "status": PipelineStatus.PENDING.value,
            "contracts": len(files),
            "filenames": [f.filename for f in files],
        }

    except HTTPException:
        raise
    except Exception as e:
        # Cleanup on error
        for path in temp_paths:
            if path.exists():
                path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text")
async def process_text(
    request: PipelineTextRequest,
) -> dict[str, Any]:
    """
    Process contract text directly (synchronous).
    """
    orchestrator = get_pipeline_orchestrator()

    try:
        result = await orchestrator.process_contract_text(
            text=request.text,
            cuad_id=request.cuad_id,
            persist=request.persist,
        )

        return result.to_dict()

    except Exception as e:
        logger.error("process_text_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{pipeline_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(
    pipeline_id: str,
) -> PipelineStatusResponse:
    """
    Get pipeline execution status.
    """
    cache = get_redis_cache()

    status = cache.get_contract_status(pipeline_id)
    if not status:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    result = cache.get(f"pipeline:{pipeline_id}:result")

    return PipelineStatusResponse(
        pipeline_id=pipeline_id,
        status=status,
        entities=result.get("entities", 0) if result else 0,
        triples=result.get("triples", 0) if result else 0,
        error=result.get("error") if result else None,
        started_at=result.get("started_at") if result else None,
        completed_at=result.get("completed_at") if result else None,
        duration_seconds=result.get("duration_seconds") if result else None,
    )


@router.get("/result/{pipeline_id}")
async def get_pipeline_result(
    pipeline_id: str,
) -> dict[str, Any]:
    """
    Get full pipeline result.
    """
    cache = get_redis_cache()

    result = cache.get(f"pipeline:{pipeline_id}:result")
    if not result:
        raise HTTPException(status_code=404, detail="Pipeline result not found")

    return result


@router.post("/single/{contract_path:path}")
async def process_single_contract(
    contract_path: str,
    persist: bool = True,
) -> dict[str, Any]:
    """
    Process a single contract (synchronous).
    """
    orchestrator = get_pipeline_orchestrator()

    path = Path(contract_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Contract file not found")

    try:
        result = await orchestrator.process_single_contract(path, persist=persist)
        return result.to_dict()

    except Exception as e:
        logger.error("process_single_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate/{pipeline_id}")
async def validate_pipeline_result(
    pipeline_id: str,
) -> dict[str, Any]:
    """
    Validate pipeline result.
    """
    orchestrator = get_pipeline_orchestrator()
    cache = get_redis_cache()

    result_data = cache.get(f"pipeline:{pipeline_id}:result")
    if not result_data:
        raise HTTPException(status_code=404, detail="Pipeline result not found")

    # This would require reconstructing the result object
    # For now, return cached data
    return {
        "pipeline_id": pipeline_id,
        "valid": result_data.get("status") == "completed",
        "entities": result_data.get("entities", 0),
        "triples": result_data.get("triples", 0),
    }
