"""
ML Pipeline Orchestrator for Ceiling Component Placement.

This module implements the core pipeline logic:
1. Sequential execution of placement stages
2. Parallel execution of competing models within a stage
3. Model selection based on confidence and validation
4. Result aggregation and grid updates
"""
import time
import copy
import logging
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from src.models.data_models import (
    CeilingGrid, ComponentType, PlacementResult, 
    PipelineResult, CellStatus, ComponentPlacement
)
from src.models.placement_models import (
    BasePlacementModel, ModelConfig, create_model
)

logger = logging.getLogger(__name__)


class SelectionStrategy(str, Enum):
    """Strategy for selecting the best model result."""
    HIGHEST_CONFIDENCE = "highest_confidence"
    MOST_PLACEMENTS = "most_placements"
    WEIGHTED_SCORE = "weighted_score"


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    stage_name: str
    component_type: ComponentType
    model_configs: List[ModelConfig]
    selection_strategy: SelectionStrategy = SelectionStrategy.HIGHEST_CONFIDENCE
    min_confidence_threshold: float = 0.5
    parallel_execution: bool = True
    max_workers: int = 4


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""
    pipeline_id: str
    pipeline_version: str
    stages: List[StageConfig]
    fail_on_invalid: bool = False
    track_metrics: bool = True


class ModelSelector:
    """
    Selects the best result from multiple model outputs.
    """
    
    @staticmethod
    def select_best(
        results: List[PlacementResult],
        strategy: SelectionStrategy,
        min_confidence: float = 0.5
    ) -> Optional[PlacementResult]:
        """
        Select the best result based on the given strategy.
        
        Args:
            results: List of results from competing models
            strategy: Selection strategy to use
            min_confidence: Minimum confidence threshold
            
        Returns:
            The best PlacementResult, or None if no valid results
        """
        # Filter to valid results above threshold
        valid_results = [
            r for r in results 
            if r.is_valid and r.confidence_score >= min_confidence
        ]
        
        if not valid_results:
            # Fall back to any valid result
            valid_results = [r for r in results if r.is_valid]
        
        if not valid_results:
            logger.warning("No valid results found, returning highest confidence invalid result")
            valid_results = results
        
        if not valid_results:
            return None
        
        if strategy == SelectionStrategy.HIGHEST_CONFIDENCE:
            return max(valid_results, key=lambda r: r.confidence_score)
        
        elif strategy == SelectionStrategy.MOST_PLACEMENTS:
            return max(valid_results, key=lambda r: len(r.placements))
        
        elif strategy == SelectionStrategy.WEIGHTED_SCORE:
            # Weighted combination of confidence and placement count
            def weighted_score(r: PlacementResult) -> float:
                return r.confidence_score * 0.7 + (len(r.placements) / 100) * 0.3
            return max(valid_results, key=weighted_score)
        
        return valid_results[0]


class PipelineStage:
    """
    Represents a single stage in the ML pipeline.
    
    Each stage:
    - Runs one or more models for a specific component type
    - Selects the best result
    - Updates the grid with placements
    """
    
    def __init__(self, config: StageConfig):
        self.config = config
        self.models: List[BasePlacementModel] = []
        self.selector = ModelSelector()
        self._is_initialized = False
    
    def initialize(self) -> None:
        """Initialize and load all models for this stage."""
        logger.info(f"Initializing stage: {self.config.stage_name}")
        
        for model_config in self.config.model_configs:
            model = create_model(model_config)
            model.load()
            self.models.append(model)
            logger.info(f"  Loaded model: {model_config.model_id}")
        
        self._is_initialized = True
    
    def execute(self, grid: CeilingGrid) -> tuple[List[PlacementResult], PlacementResult]:
        """
        Execute all models and select the best result.
        
        Args:
            grid: Current state of the ceiling grid
            
        Returns:
            Tuple of (all_results, selected_best_result)
        """
        if not self._is_initialized:
            raise RuntimeError("Stage not initialized. Call initialize() first.")
        
        all_results = []
        
        if self.config.parallel_execution and len(self.models) > 1:
            # Run models in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(model.predict, grid): model 
                    for model in self.models
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        model = futures[future]
                        logger.error(f"Model {model.config.model_id} failed: {e}")
        else:
            # Run models sequentially
            for model in self.models:
                try:
                    result = model.predict(grid)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Model {model.config.model_id} failed: {e}")
        
        # Select the best result
        best_result = self.selector.select_best(
            all_results,
            self.config.selection_strategy,
            self.config.min_confidence_threshold
        )
        
        return all_results, best_result


class CeilingMLPipeline:
    """
    Main ML Pipeline orchestrator for ceiling component placement.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages: List[PipelineStage] = []
        self._is_initialized = False
        self._metrics: Dict[str, any] = {}
    
    def initialize(self) -> None:
        """Initialize all pipeline stages."""
        logger.info(f"Initializing pipeline: {self.config.pipeline_id}")
        
        for stage_config in self.config.stages:
            stage = PipelineStage(stage_config)
            stage.initialize()
            self.stages.append(stage)
        
        self._is_initialized = True
        logger.info(f"Pipeline initialized with {len(self.stages)} stages")
    
    def _apply_placements(
        self, 
        grid: CeilingGrid, 
        result: PlacementResult
    ) -> CeilingGrid:
        """Apply placements to the grid, returning a new grid."""
        # Deep copy to avoid mutating the original
        new_grid = copy.deepcopy(grid)
        
        for placement in result.placements:
            cell = new_grid.cells[placement.row][placement.col]
            if cell.status == CellStatus.VALID:
                cell.status = CellStatus.OCCUPIED
                cell.component = placement.component_type
                cell.metadata["model_id"] = placement.model_id
                cell.metadata["confidence"] = placement.confidence
        
        return new_grid
    
    def execute(self, input_grid: CeilingGrid) -> PipelineResult:
        """
        Execute the full pipeline on the input grid.
        
        Args:
            input_grid: The initial ceiling grid
            
        Returns:
            PipelineResult containing the completed building data
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        current_grid = copy.deepcopy(input_grid)
        stage_results: Dict[str, List[PlacementResult]] = {}
        selected_results: Dict[str, PlacementResult] = {}
        
        logger.info(f"Starting pipeline execution for room: {input_grid.room_id}")
        
        for stage in self.stages:
            stage_name = stage.config.stage_name
            logger.info(f"Executing stage: {stage_name}")
            
            # Execute stage with current grid state
            all_results, best_result = stage.execute(current_grid)
            
            stage_results[stage_name] = all_results
            
            if best_result:
                selected_results[stage_name] = best_result
                
                # Update grid with placements for next stage
                current_grid = self._apply_placements(current_grid, best_result)
                
                logger.info(
                    f"  Selected: {best_result.model_id} "
                    f"(confidence: {best_result.confidence_score:.2f}, "
                    f"placements: {len(best_result.placements)})"
                )
            else:
                logger.warning(f"  No valid result for stage: {stage_name}")
                if self.config.fail_on_invalid:
                    raise RuntimeError(f"Stage {stage_name} produced no valid results")
        
        total_time = (time.time() - start_time) * 1000
        
        result = PipelineResult(
            room_id=input_grid.room_id,
            input_grid=input_grid,
            output_grid=current_grid,
            stage_results=stage_results,
            selected_results=selected_results,
            total_execution_time_ms=total_time,
            pipeline_version=self.config.pipeline_version,
            metadata={
                "pipeline_id": self.config.pipeline_id,
                "stages_executed": len(self.stages),
                "total_models_run": sum(
                    len(results) for results in stage_results.values()
                )
            }
        )
        
        logger.info(f"Pipeline completed in {total_time:.2f}ms")
        
        return result


def create_default_pipeline() -> CeilingMLPipeline:
    """
    Create a pipeline with default configuration.
    
    This demonstrates the typical setup with multiple models
    for certain stages (like lighting) and single models for others.
    """
    config = PipelineConfig(
        pipeline_id="ceiling_pipeline_v1",
        pipeline_version="1.0.0",
        stages=[
            # Stage 1: Light placement with competing models
            StageConfig(
                stage_name="light_placement",
                component_type=ComponentType.LIGHT,
                model_configs=[
                    ModelConfig(
                        model_id="light_placement_v1",
                        model_version="1.0.0",
                        component_type=ComponentType.LIGHT,
                        parameters={"spacing": 3}
                    ),
                    ModelConfig(
                        model_id="light_placement_v2",
                        model_version="1.0.0",
                        component_type=ComponentType.LIGHT,
                        parameters={"spacing": 4}
                    ),
                    ModelConfig(
                        model_id="light_placement_v3",
                        model_version="1.0.0",
                        component_type=ComponentType.LIGHT,
                        parameters={}
                    ),
                    ModelConfig(
                        model_id="light_placement_v4",
                        model_version="1.0.0",
                        component_type=ComponentType.LIGHT,
                        parameters={}
                    ),
                ],
                selection_strategy=SelectionStrategy.HIGHEST_CONFIDENCE,
                parallel_execution=True
            ),
            # Stage 2: Air supply placement
            StageConfig(
                stage_name="air_supply_placement",
                component_type=ComponentType.AIR_SUPPLY,
                model_configs=[
                    ModelConfig(
                        model_id="air_supply_v1",
                        model_version="1.0.0",
                        component_type=ComponentType.AIR_SUPPLY,
                        parameters={"spacing": 5}
                    ),
                ],
                selection_strategy=SelectionStrategy.HIGHEST_CONFIDENCE,
                parallel_execution=False
            ),
            # Stage 3: Air return placement
            StageConfig(
                stage_name="air_return_placement",
                component_type=ComponentType.AIR_RETURN,
                model_configs=[
                    ModelConfig(
                        model_id="air_return_v1",
                        model_version="1.0.0",
                        component_type=ComponentType.AIR_RETURN,
                        parameters={}
                    ),
                ],
                selection_strategy=SelectionStrategy.HIGHEST_CONFIDENCE,
                parallel_execution=False
            ),
            # Stage 4: Smoke detector placement
            StageConfig(
                stage_name="smoke_detector_placement",
                component_type=ComponentType.SMOKE_DETECTOR,
                model_configs=[
                    ModelConfig(
                        model_id="smoke_detector_v1",
                        model_version="1.0.0",
                        component_type=ComponentType.SMOKE_DETECTOR,
                        parameters={"spacing": 6}
                    ),
                ],
                selection_strategy=SelectionStrategy.HIGHEST_CONFIDENCE,
                parallel_execution=False
            ),
        ],
        fail_on_invalid=False,
        track_metrics=True
    )
    
    pipeline = CeilingMLPipeline(config)
    return pipeline
