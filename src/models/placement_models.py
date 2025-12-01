"""
Base model interface and. implementations for ceiling component placement.
This module defines the contract that all ML models must implement, 
plus mock implementations for demnostration purposes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
import random
import json
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
import requests as http_requests

from src.models.data_models import (CeilingGrid, ComponentType,
                                    PlacementResult, ComponentPlacement, 
                                    CellStatus)

@dataclass
class ModelConfig:
    """Configuration for a placement model."""
    model_id: str
    model_version: str
    component_type: ComponentType
    model_path: Optional[str] = None # Path to model file if applicable
    mlflow_uri: Optional[str] = None  # MLflow tracking URI if applicable
    parameters: Dict[str, Any] = None  # Additional model-specific parameters

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

class BasePlacementModel(ABC):
    """
    Abstract base class for all component placement models.

    All models that participate in the pipeline must implement this interface.
    This allows for easy swapping of models and A/B testing
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None 
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def predict(self, grid: CeilingGrid) -> PlacementResult:
        """
        Generate placement predictions for the given ceiling grid.

        Args:
            grid: The input ceiling grid (may already have some components placed).

        Returns:
            PlacementResult containing predicted placements.

        """

        pass

    def validate_placements(
            self,
            grid: CeilingGrid,
            placements: List[ComponentPlacement]) -> tuple[bool, List[str]]:
        
        """
        Validates the placements.

        Args:
            grid: The ceiling grid
            placements: List of proposed placements

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for placement in placements:
            # Check bounds
            if placement.row < 0 or placement.row >= grid.height:
                errors.append(f"Placement at ({placement.row}, {placement.col}) is out of bounds (row)")
            
            if placement.col < 0 or placement.col >= grid.width:
                errors.append(f"Placement at ({placement.row}, {placement.col}) is out of bounds (col)")
                continue    

            cell = grid.cells[placement.row][placement.col]

            # Check if cell is valid for placement
            if cell.status == CellStatus.INVALID:
                errors.append(f"Cannot place at ({placement.row}, {placement.col}): Cell is invalid")
            elif cell.status == CellStatus.OCCUPIED:
                errors.append(f"Cannot place at ({placement.row}, {placement.col}): Cell is already occupied")
        
        return (len(errors) == 0, errors)
    
    @property
    def is_loaded(self) -> bool:
        """Indicates if the model is loaded into memory."""
        return self._is_loaded
    

# ============================================================================
# Mock Model Implementations (for demonstration)
# In production, these would load real ML models from MLflow registry/scikit-learn or similar
# ============================================================================

class MockLightPlacementModelV1(BasePlacementModel):
    """
    Mock model for light placement - Version 1 (Grid-based).
    Places lights in a regular grid pattern.
    """
    
    def load(self) -> None:
        # In production: self.model = mlflow.pytorch.load_model(self.config.model_path)
        self._is_loaded = True
    
    def predict(self, grid: CeilingGrid) -> PlacementResult:
        start_time = time.time()
        
        placements = []
        spacing = self.config.parameters.get("spacing", 3)
        
        for row in range(0, grid.height, spacing):
            for col in range(0, grid.width, spacing):
                if row < grid.height and col < grid.width:
                    cell = grid.cells[row][col]
                    if cell.status == CellStatus.VALID:
                        placements.append(ComponentPlacement(
                            component_type=ComponentType.LIGHT,
                            row=row,
                            col=col,
                            confidence=0.85 + random.uniform(0, 0.1),
                            model_id=self.config.model_id
                        ))
        
        is_valid, errors = self.validate_placements(grid, placements)
        execution_time = (time.time() - start_time) * 1000
        
        return PlacementResult(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            component_type=ComponentType.LIGHT,
            placements=placements,
            execution_time_ms=execution_time,
            confidence_score=0.88,
            is_valid=is_valid,
            validation_errors=errors
        )


class MockLightPlacementModelV2(BasePlacementModel):
    """
    Mock model for light placement - Version 2.
    """
    
    def load(self) -> None:
        self._is_loaded = True
    
    def predict(self, grid: CeilingGrid) -> PlacementResult:
        start_time = time.time()
        
        placements = []
        # Simulate a smarter algorithm with offset rows
        spacing = self.config.parameters.get("spacing", 4)
        
        for row in range(1, grid.height - 1, spacing):
            offset = (row // spacing) % 2 * (spacing // 2)
            for col in range(1 + offset, grid.width - 1, spacing):
                cell = grid.cells[row][col]
                if cell.status == CellStatus.VALID:
                    placements.append(ComponentPlacement(
                        component_type=ComponentType.LIGHT,
                        row=row,
                        col=col,
                        confidence=0.90 + random.uniform(0, 0.08),
                        model_id=self.config.model_id
                    ))
        
        is_valid, errors = self.validate_placements(grid, placements)
        execution_time = (time.time() - start_time) * 1000
        
        return PlacementResult(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            component_type=ComponentType.LIGHT,
            placements=placements,
            execution_time_ms=execution_time,
            confidence_score=0.92,
            is_valid=is_valid,
            validation_errors=errors
        )


class MockLightPlacementModelV3(BasePlacementModel):
    """
    Light placement using K-Means clustering (scikit-learn demo).
    """
    
    def load(self) -> None:
        self._is_loaded = True
    
    def predict(self, grid: CeilingGrid) -> PlacementResult:
        start_time = time.time()
        
        # Get valid cell coordinates
        coords = np.array([[r, c] for r in range(grid.height) for c in range(grid.width)
                          if grid.cells[r][c].status == CellStatus.VALID])
        
        placements = []
        if len(coords) > 0:
            n_clusters = max(1, len(coords) // 9)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(coords)
            
            for center in kmeans.cluster_centers_:
                idx = np.argmin(np.sum((coords - center) ** 2, axis=1))
                row, col = int(coords[idx, 0]), int(coords[idx, 1])
                placements.append(ComponentPlacement(
                    component_type=ComponentType.LIGHT,
                    row=row, col=col, confidence=0.91, model_id=self.config.model_id
                ))
        
        is_valid, errors = self.validate_placements(grid, placements)
        
        return PlacementResult(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            component_type=ComponentType.LIGHT,
            placements=placements,
            execution_time_ms=(time.time() - start_time) * 1000,
            confidence_score=0.91,
            is_valid=is_valid,
            validation_errors=errors
        )


class MockLightPlacementModelV4(BasePlacementModel):
    """
    Light placement using Ollama LLM (qwen2.5:0.5b).
    """
    
    def load(self) -> None:
        self._is_loaded = True
    
    def predict(self, grid: CeilingGrid) -> PlacementResult:
        start_time = time.time()
        
        matrix = grid.to_matrix()
        
        prompt = f"""Given this {grid.height}x{grid.width} ceiling grid as JSON:
{json.dumps(matrix)}

Where: 0=valid cell, 1=invalid cell, 2+=already occupied.
Return optimal light positions for even coverage. Only use cells with value 0.
Respond with ONLY a JSON array of [row,col] pairs. Example: [[1,2],[3,4]]"""

        placements = []
        try:
            response = http_requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "qwen2.5:0.5b", "prompt": prompt, "stream": False},
                timeout=30
            )
            result = response.json().get("response", "[]")

            # print(f"Input grid {json.dumps(matrix)}")

            # print(f"LLM Response: {result}")
            
            # Parse JSON array from response
            start_idx = result.find("[")
            end_idx = result.rfind("]") + 1
            if start_idx != -1 and end_idx > start_idx:
                positions = json.loads(result[start_idx:end_idx])
                for pos in positions:
                    if len(pos) >= 2:
                        row, col = int(pos[0]), int(pos[1])
                        if 0 <= row < grid.height and 0 <= col < grid.width:
                            if grid.cells[row][col].status == CellStatus.VALID:
                                placements.append(ComponentPlacement(
                                    component_type=ComponentType.LIGHT,
                                    row=row, col=col, confidence=0.85, model_id=self.config.model_id
                                ))
        except Exception:
            pass  # Return empty placements on error
        
        is_valid, errors = self.validate_placements(grid, placements)

        # print(f"LLM Light Placement took {(time.time() - start_time) * 1000:.2f} ms for {len(placements)} placements.")
        
        return PlacementResult(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            component_type=ComponentType.LIGHT,
            placements=placements,
            execution_time_ms=(time.time() - start_time) * 1000,
            confidence_score=0.85,
            is_valid=is_valid,
            validation_errors=errors
        )


class MockAirSupplyModel(BasePlacementModel):
    """Mock model for air supply point placement."""
    
    def load(self) -> None:
        self._is_loaded = True
    
    def predict(self, grid: CeilingGrid) -> PlacementResult:
        start_time = time.time()
        
        placements = []
        # Place air supply points considering existing lights
        spacing = self.config.parameters.get("spacing", 5)
        
        for row in range(2, grid.height - 2, spacing):
            for col in range(2, grid.width - 2, spacing):
                cell = grid.cells[row][col]
                if cell.status == CellStatus.VALID and cell.component is None:
                    placements.append(ComponentPlacement(
                        component_type=ComponentType.AIR_SUPPLY,
                        row=row,
                        col=col,
                        confidence=0.87,
                        model_id=self.config.model_id
                    ))
        
        is_valid, errors = self.validate_placements(grid, placements)
        execution_time = (time.time() - start_time) * 1000
        
        return PlacementResult(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            component_type=ComponentType.AIR_SUPPLY,
            placements=placements,
            execution_time_ms=execution_time,
            confidence_score=0.87,
            is_valid=is_valid,
            validation_errors=errors
        )


class MockAirReturnModel(BasePlacementModel):
    """Mock model for air return point placement."""
    
    def load(self) -> None:
        self._is_loaded = True
    
    def predict(self, grid: CeilingGrid) -> PlacementResult:
        start_time = time.time()
        
        placements = []
        # Place air returns at room edges
        for row in [1, grid.height - 2]:
            for col in range(2, grid.width - 2, 6):
                if row < grid.height and col < grid.width:
                    cell = grid.cells[row][col]
                    if cell.status == CellStatus.VALID and cell.component is None:
                        placements.append(ComponentPlacement(
                            component_type=ComponentType.AIR_RETURN,
                            row=row,
                            col=col,
                            confidence=0.85,
                            model_id=self.config.model_id
                        ))
        
        is_valid, errors = self.validate_placements(grid, placements)
        execution_time = (time.time() - start_time) * 1000
        
        return PlacementResult(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            component_type=ComponentType.AIR_RETURN,
            placements=placements,
            execution_time_ms=execution_time,
            confidence_score=0.85,
            is_valid=is_valid,
            validation_errors=errors
        )


class MockSmokeDetectorModel(BasePlacementModel):
    """Mock model for smoke detector placement."""
    
    def load(self) -> None:
        self._is_loaded = True
    
    def predict(self, grid: CeilingGrid) -> PlacementResult:
        start_time = time.time()
        
        placements = []
        spacing = self.config.parameters.get("spacing", 6)
        
        for row in range(spacing // 2, grid.height, spacing):
            for col in range(spacing // 2, grid.width, spacing):
                if row < grid.height and col < grid.width:
                    cell = grid.cells[row][col]
                    if cell.status == CellStatus.VALID and cell.component is None:
                        placements.append(ComponentPlacement(
                            component_type=ComponentType.SMOKE_DETECTOR,
                            row=row,
                            col=col,
                            confidence=0.95,
                            model_id=self.config.model_id
                        ))
        
        is_valid, errors = self.validate_placements(grid, placements)
        execution_time = (time.time() - start_time) * 1000
        
        return PlacementResult(
            model_id=self.config.model_id,
            model_version=self.config.model_version,
            component_type=ComponentType.SMOKE_DETECTOR,
            placements=placements,
            execution_time_ms=execution_time,
            confidence_score=0.95,
            is_valid=is_valid,
            validation_errors=errors
        )


# Model Registry - Maps model IDs to their classes
MODEL_REGISTRY = {
    "light_placement_v1": MockLightPlacementModelV1,
    "light_placement_v2": MockLightPlacementModelV2,
    "light_placement_v3": MockLightPlacementModelV3,
    "light_placement_v4": MockLightPlacementModelV4,
    "air_supply_v1": MockAirSupplyModel,
    "air_return_v1": MockAirReturnModel,
    "smoke_detector_v1": MockSmokeDetectorModel,
}


def create_model(config: ModelConfig) -> BasePlacementModel:
    """Factory function to create model instances."""
    model_class = MODEL_REGISTRY.get(config.model_id)
    if model_class is None:
        raise ValueError(f"Unknown model ID: {config.model_id}")
    return model_class(config)