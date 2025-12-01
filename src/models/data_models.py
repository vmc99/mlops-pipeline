"""
Data models for ceiling grid representation and component placement.
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from uuid import uuid4

class ComponentType(str, Enum):
    """Types of components that can be placed on the ceiling grid."""
    LIGHT = "light"
    AIR_SUPPLY = "air_supply"
    AIR_RETURN = "air_return"
    SMOKE_DETECTOR = "smoke_detector"


class CellStatus(str, Enum):
    """Status of a ceiling grid."""
    VALID = "valid"          # Can place component
    INVALID = "invalid"      # Cannot place component
    OCCUPIED = "occupied"    # Already has a component


class GridCell(BaseModel):
    """Represents a single cell in the ceiling grid."""
    row: int
    column: int
    status: CellStatus = CellStatus.VALID
    component: Optional[ComponentType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CeilingGrid(BaseModel):
    """
    Represents the entire ceiling grid for a room.
    This is the primary input to the ML pipeline.
    """
    room_id: str = Field(default_factory=lambda: str(uuid4()))
    width: int  # number of columns
    height: int # number of rows
    cell_size: float = 2.0 # size of each cell in feet
    cells: List[List[GridCell]]
    room_metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_matrix(cls, matrix: List[List[int]], room_id: str = None) -> "CeilingGrid":
        """
        create a CeilingGrid from a simple 2D interger matrix representation.
        0 -> valid, 1 -> invalid, >=2 -> occupied with component type
        """
        height = len(matrix)
        width = len(matrix[0]) if height > 0 else 0
        
        cells = []
        for row_idx, row in enumerate(matrix):
            cell_row = []
            for col_idx, value in enumerate(row):
                if value == 0:
                    status = CellStatus.VALID
                    component = None
                elif value == 1:
                    status = CellStatus.INVALID
                    component = None
                else:
                    status = CellStatus.OCCUPIED
                    component_map = {
                        2: ComponentType.LIGHT,
                        3: ComponentType.AIR_SUPPLY,
                        4: ComponentType.AIR_RETURN,
                        5: ComponentType.SMOKE_DETECTOR
                    }
                    component = component_map.get(value)

                cell_row.append(GridCell(
                    row=row_idx,
                    column=col_idx,
                    status=status,
                    component=component
                ))
            cells.append(cell_row)

        return cls(
            room_id = room_id or str(uuid4()),
            width = width,
            height = height,
            cells = cells # 
        )
    
    def to_matrix(self) -> List[List[int]]:
        """ Convert back to 2D integer matrix representation."""
        component_to_int = {
            None: 0,
            ComponentType.LIGHT: 2,
            ComponentType.AIR_SUPPLY: 3,
            ComponentType.AIR_RETURN: 4,
            ComponentType.SMOKE_DETECTOR: 5
        }

        matrix = []
        for row in self.cells:
            matrix_row = []
            for cell in row:
                if cell.status == CellStatus.INVALID:
                    matrix_row.append(1)
                else:
                    matrix_row.append(component_to_int[cell.component])
            
            matrix.append(matrix_row)
        
        return matrix

class ComponentPlacement(BaseModel):
    """Represents a single component placement decision."""
    
    component_type: ComponentType
    row: int
    col: int
    confidence: float = Field(ge=0.0, le=1.0)
    model_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PlacementResult(BaseModel):
    """Result from a single model's placement prediction."""

    model_id: str
    model_version: str
    component_type: ComponentType
    placements: List[ComponentPlacement]
    execution_time_ms: float
    confidence_score: float = Field(ge=0.0, le=1.0)
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PipelineResult(BaseModel):
    """
    Final output of the ML pipeline. (Complete JSON)
    """
    room_id: str
    input_grid: CeilingGrid
    output_grid: CeilingGrid
    stage_results: Dict[str, List[PlacementResult]] # Stage name -> results from all models
    selected_results: Dict[str, PlacementResult] # stage name -> best selected result
    total_execution_time_ms: float
    pipeline_version: str
    metadata: Dict[str, Any] = Field(default_factory=dict)