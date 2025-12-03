"""
Tests for the Ceiling ML Pipeline.

Run with: pytest tests/test_pipeline.py -v 2>&1
"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.data_models import (
    CeilingGrid, ComponentType, CellStatus, GridCell
)
from src.models.placement_models import (
    ModelConfig, create_model, MockLightPlacementModelV1
)
from src.pipeline.orchestrator import (
    CeilingMLPipeline, create_default_pipeline, 
    PipelineStage, StageConfig, SelectionStrategy
)


class TestCeilingGrid:
    """Tests for CeilingGrid data model."""
    
    def test_from_matrix_basic(self):
        """Test creating grid from simple matrix."""
        matrix = [
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
        ]
        grid = CeilingGrid.from_matrix(matrix)
        
        assert grid.width == 3
        assert grid.height == 3
        assert grid.cells[0][0].status == CellStatus.VALID
        assert grid.cells[0][2].status == CellStatus.INVALID
        assert grid.cells[2][0].status == CellStatus.INVALID
    
    def test_from_matrix_with_components(self):
        """Test creating grid with existing components."""
        matrix = [
            [0, 2, 0],  # 2 = light
            [0, 0, 3],  # 3 = air supply
            [5, 0, 0]   # 5 = smoke detector
        ]
        grid = CeilingGrid.from_matrix(matrix)
        
        assert grid.cells[0][1].component == ComponentType.LIGHT
        assert grid.cells[1][2].component == ComponentType.AIR_SUPPLY
        assert grid.cells[2][0].component == ComponentType.SMOKE_DETECTOR
    
    def test_to_matrix_roundtrip(self):
        """Test that to_matrix reverses from_matrix."""
        original_matrix = [
            [0, 0, 1, 0],
            [0, 2, 0, 0],
            [1, 0, 0, 5],
            [0, 0, 0, 0]
        ]
        grid = CeilingGrid.from_matrix(original_matrix)
        result_matrix = grid.to_matrix()
        
        assert result_matrix == original_matrix


class TestPlacementModels:
    """Tests for placement models."""
    
    def test_create_model(self):
        """Test model factory function."""
        config = ModelConfig(
            model_id="light_placement_v1",
            model_version="1.0.0",
            component_type=ComponentType.LIGHT,
            parameters={"spacing": 3}
        )
        model = create_model(config)
        
        assert isinstance(model, MockLightPlacementModelV1)
        assert model.config.model_id == "light_placement_v1"
    
    def test_model_predict(self):
        """Test model prediction."""
        config = ModelConfig(
            model_id="light_placement_v1",
            model_version="1.0.0",
            component_type=ComponentType.LIGHT,
            parameters={"spacing": 2}
        )
        model = create_model(config)
        model.load()
        
        matrix = [[0] * 6 for _ in range(6)]
        grid = CeilingGrid.from_matrix(matrix)
        
        result = model.predict(grid)
        
        assert result.model_id == "light_placement_v1"
        assert result.component_type == ComponentType.LIGHT
        assert len(result.placements) > 0
        assert result.is_valid
    
    def test_model_validates_placements(self):
        """Test that model validates invalid placements."""
        config = ModelConfig(
            model_id="light_placement_v1",
            model_version="1.0.0",
            component_type=ComponentType.LIGHT,
            parameters={"spacing": 2}
        )
        model = create_model(config)
        model.load()
        
        # Grid with invalid cells
        matrix = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        grid = CeilingGrid.from_matrix(matrix)
        
        result = model.predict(grid)
        
        # Should only place in the one valid cell
        assert result.is_valid


class TestPipelineStage:
    """Tests for pipeline stages."""
    
    def test_stage_initialization(self):
        """Test stage initializes models correctly."""
        stage_config = StageConfig(
            stage_name="test_stage",
            component_type=ComponentType.LIGHT,
            model_configs=[
                ModelConfig(
                    model_id="light_placement_v1",
                    model_version="1.0.0",
                    component_type=ComponentType.LIGHT
                ),
                ModelConfig(
                    model_id="light_placement_v2",
                    model_version="1.0.0",
                    component_type=ComponentType.LIGHT
                )
            ],
            selection_strategy=SelectionStrategy.HIGHEST_CONFIDENCE
        )
        
        stage = PipelineStage(stage_config)
        stage.initialize()
        
        assert len(stage.models) == 2
        assert all(m.is_loaded for m in stage.models)
    
    def test_stage_execution(self):
        """Test stage executes and selects best model."""
        stage_config = StageConfig(
            stage_name="test_stage",
            component_type=ComponentType.LIGHT,
            model_configs=[
                ModelConfig(
                    model_id="light_placement_v1",
                    model_version="1.0.0",
                    component_type=ComponentType.LIGHT
                ),
                ModelConfig(
                    model_id="light_placement_v2",
                    model_version="1.0.0",
                    component_type=ComponentType.LIGHT
                )
            ],
            parallel_execution=True
        )
        
        stage = PipelineStage(stage_config)
        stage.initialize()
        
        matrix = [[0] * 10 for _ in range(10)]
        grid = CeilingGrid.from_matrix(matrix)
        
        all_results, best_result = stage.execute(grid)
        
        assert len(all_results) == 2
        assert best_result is not None
        assert best_result.model_id in ["light_placement_v1", "light_placement_v2"]


class TestCeilingMLPipeline:
    """Tests for the full pipeline."""
    
    def test_default_pipeline_creation(self):
        """Test creating default pipeline."""
        pipeline = create_default_pipeline()
        
        assert pipeline.config.pipeline_id == "ceiling_pipeline_v1"
        assert len(pipeline.config.stages) == 4
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = create_default_pipeline()
        pipeline.initialize()
        
        assert pipeline._is_initialized
        assert len(pipeline.stages) == 4
    
    def test_pipeline_execution(self):
        """Test full pipeline execution."""
        pipeline = create_default_pipeline()
        pipeline.initialize()
        
        # Create a test grid
        matrix = [[0] * 12 for _ in range(10)]
        # Add some invalid cells
        matrix[2][3] = 1
        matrix[2][4] = 1
        matrix[3][3] = 1
        matrix[3][4] = 1
        
        grid = CeilingGrid.from_matrix(matrix, room_id="test-room")
        
        result = pipeline.execute(grid)
        
        assert result.room_id == "test-room"
        assert len(result.selected_results) == 4
        assert result.total_execution_time_ms > 0
        
        # Check that components were placed
        output_matrix = result.output_grid.to_matrix()
        placed_components = sum(
            1 for row in output_matrix for cell in row if cell > 1
        )
        assert placed_components > 0
    
    def test_pipeline_preserves_invalid_cells(self):
        """Test that pipeline doesn't place in invalid cells."""
        pipeline = create_default_pipeline()
        pipeline.initialize()
        
        matrix = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        grid = CeilingGrid.from_matrix(matrix)
        
        result = pipeline.execute(grid)
        output = result.output_grid.to_matrix()
        
        # Check corners still invalid
        assert output[0][0] == 1
        assert output[0][3] == 1
        assert output[3][0] == 1
        assert output[3][3] == 1


# ============================================================================
# API Tests (requires running server or test client)
# ============================================================================

@pytest.fixture
def test_client():
    """Create a test client for API testing."""
    from fastapi.testclient import TestClient
    import src.api.main as main_module
    from src.pipeline.orchestrator import create_default_pipeline
    
    # Initialize pipeline for testing (update the module's global variable)
    if main_module.pipeline is None or not main_module.pipeline._is_initialized:
        main_module.pipeline = create_default_pipeline()
        main_module.pipeline.initialize()
    
    return TestClient(main_module.app)


class TestAPI:
    """Tests for the FastAPI endpoints."""
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "pipeline_initialized" in data
    
    def test_run_pipeline_endpoint(self, test_client):
        """Test pipeline execution endpoint."""
        input_data = {
            "room_id": "test-room",
            "width": 6,
            "height": 6,
            "matrix": [[0] * 6 for _ in range(6)],
            "cell_size": 2.0
        }
        
        response = test_client.post("/pipeline/run", json=input_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["room_id"] == "test-room"
        assert "output_matrix" in data
        assert "execution_time_ms" in data
        assert "stage_summary" in data
    
    def test_list_stages_endpoint(self, test_client):
        """Test pipeline stages listing endpoint."""
        response = test_client.get("/pipeline/stages")
        assert response.status_code == 200
        
        data = response.json()
        assert "stages" in data
        assert len(data["stages"]) == 4
    
    def test_list_models_endpoint(self, test_client):
        """Test models listing endpoint."""
        response = test_client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "available_models" in data
        assert "light_placement_v1" in data["available_models"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
