"""
Interactive Streamlit UI for Ceiling ML Pipeline.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Cell type mappings
CELL_TYPES = {
    0: ("Valid", ""),
    1: ("Invalid", ""),
    2: ("Light", "L"),
    3: ("Air Supply", "S"),
    4: ("Air Return", "R"),
    5: ("Smoke Detector", "D"),
}

CELL_COLORS = {
    0: "#90EE90",  # valid
    1: "#404040",  # invalid
    2: "#FFD700",  # light
    3: "#4169E1",  # air supply
    4: "#DC143C",  # air return
    5: "#FF6347",  # smoke detector
}


def init_session_state():
    """Initialize session state variables."""
    if "width" not in st.session_state:
        st.session_state.width = 10
    if "height" not in st.session_state:
        st.session_state.height = 8
    if "grid" not in st.session_state:
        st.session_state.grid = create_empty_grid(st.session_state.width, st.session_state.height)
    if "result" not in st.session_state:
        st.session_state.result = None
    if "room_id" not in st.session_state:
        st.session_state.room_id = "room-001"
    if "editor_key" not in st.session_state:
        st.session_state.editor_key = 0


def update_grid_from_editor():
    """Callback to update grid from editor changes."""
    editor_key = f"grid_editor_{st.session_state.editor_key}"
    if editor_key in st.session_state:
        edited_data = st.session_state[editor_key]
        if "edited_rows" in edited_data:
            for row_idx, changes in edited_data["edited_rows"].items():
                row_idx = int(row_idx)
                for col_idx, value in changes.items():
                    col_idx = int(col_idx)
                    if value is not None:
                        try:
                            st.session_state.grid[row_idx][col_idx] = int(value)
                        except (ValueError, TypeError):
                            pass  # Ignore invalid values


def create_empty_grid(width: int, height: int) -> List[List[int]]:
    """Create an empty grid with all valid cells."""
    return [[0 for _ in range(width)] for _ in range(height)]


def grid_to_dataframe(grid: List[List[int]], show_icons: bool = True) -> pd.DataFrame:
    """Convert grid to DataFrame for display."""
    if show_icons:
        data = [[CELL_TYPES[cell][1] for cell in row] for row in grid]
    else:
        data = grid
    return pd.DataFrame(data)


def render_grid_html(grid: List[List[int]], title: str) -> str:
    """Render grid as HTML table with colors."""
    html = f"<h4>{title}</h4>"
    html += '<table style="border-collapse: collapse; margin: 10px 0;">'
    
    for row in grid:
        html += "<tr>"
        for cell in row:
            color = CELL_COLORS.get(cell, "#FFFFFF")
            icon = CELL_TYPES.get(cell, ("?", "❓"))[1]
            html += f'''<td style="
                width: 35px; 
                height: 35px; 
                background-color: {color}; 
                border: 1px solid #333;
                text-align: center;
                font-size: 18px;
            ">{icon}</td>'''
        html += "</tr>"
    
    html += "</table>"
    return html


def check_api_health() -> Dict[str, Any]:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.RequestException:
        return None


def run_pipeline(grid: List[List[int]], room_id: str) -> Dict[str, Any]:
    """Call the pipeline API."""
    payload = {
        "room_id": room_id,
        "width": len(grid[0]),
        "height": len(grid),
        "matrix": grid,
        "cell_size": 2.0
    }
    
    response = requests.post(
        f"{API_URL}/pipeline/run",
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


def main():
    st.set_page_config(
        page_title="Ceiling ML Pipeline",
        page_icon=None,
        layout="wide"
    )
    
    init_session_state()
    
    # Header
    st.title("Ceiling ML Pipeline")
    st.markdown("Interactive ceiling component placement using ML models")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Status
        health = check_api_health()
        if health and health.get("status") == "healthy":
            st.success(f"API Connected (v{health.get('pipeline_version', 'N/A')})")
            st.caption(f"Models loaded: {health.get('models_loaded', 0)}")
        else:
            st.error("API Disconnected")
            st.caption("Make sure FastAPI server is running on port 8000")
        
        st.divider()
        
        # Room Configuration
        st.subheader("Room Dimensions")
        
        new_width = st.number_input(
            "Width (cells)", 
            min_value=3, 
            max_value=50, 
            value=st.session_state.width,
            help="Number of columns in the grid"
        )
        
        new_height = st.number_input(
            "Height (cells)", 
            min_value=3, 
            max_value=50, 
            value=st.session_state.height,
            help="Number of rows in the grid"
        )
        
        st.session_state.room_id = st.text_input(
            "Room ID",
            value=st.session_state.room_id
        )
        
        if st.button("Generate New Grid", use_container_width=True):
            st.session_state.width = new_width
            st.session_state.height = new_height
            st.session_state.grid = create_empty_grid(new_width, new_height)
            st.session_state.result = None
            st.session_state.editor_key += 1  # Force new editor instance
            st.rerun()
        
        st.divider()
        
        # Legend
        st.subheader("Legend")
        for code, (name, icon) in CELL_TYPES.items():
            color = CELL_COLORS[code]
            st.markdown(
                f'<span style="background-color: {color}; padding: 2px 8px; '
                f'border-radius: 3px; margin-right: 5px;">{icon}</span> {name}',
                unsafe_allow_html=True
            )
    
    # Initialize grid if not exists
    if st.session_state.grid is None:
        st.session_state.grid = create_empty_grid(
            st.session_state.width, 
            st.session_state.height
        )
    
    # Main content - Two columns
    col1, col2 = st.columns(2)
    
    # Left column - Input Grid Editor
    with col1:
        st.header("Input Grid")
        st.caption("Edit cells: 0=Valid, 1=Invalid, 2=Light, 3=Air Supply, 4=Air Return, 5=Smoke Detector")
        
        # Create editable dataframe with dynamic key
        df = pd.DataFrame(st.session_state.grid)
        editor_key = f"grid_editor_{st.session_state.editor_key}"
        
        st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                str(i): st.column_config.NumberColumn(
                    label=str(i),
                    min_value=0,
                    max_value=5,
                    step=1,
                    format="%d"
                ) for i in range(len(df.columns))
            },
            key=editor_key,
            on_change=update_grid_from_editor
        )
        
        # Visual preview
        st.markdown("**Visual Preview:**")
        st.markdown(
            render_grid_html(st.session_state.grid, ""),
            unsafe_allow_html=True
        )
        
        # Quick actions
        st.markdown("**Quick Actions:**")
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            if st.button("Clear All", use_container_width=True):
                st.session_state.grid = create_empty_grid(
                    st.session_state.width,
                    st.session_state.height
                )
                st.session_state.editor_key += 1
                st.rerun()
        
        with action_col2:
            if st.button("Add Border", use_container_width=True):
                grid = st.session_state.grid
                h, w = len(grid), len(grid[0])
                for i in range(h):
                    grid[i][0] = 1
                    grid[i][w-1] = 1
                for j in range(w):
                    grid[0][j] = 1
                    grid[h-1][j] = 1
                st.session_state.editor_key += 1
                st.rerun()
        
        with action_col3:
            if st.button("Add Columns", use_container_width=True):
                grid = st.session_state.grid
                h, w = len(grid), len(grid[0])
                # Add 2x2 column obstacles
                for r in range(2, h-2, 4):
                    for c in range(2, w-2, 4):
                        if r+1 < h and c+1 < w:
                            grid[r][c] = 1
                            grid[r][c+1] = 1
                            grid[r+1][c] = 1
                            grid[r+1][c+1] = 1
                st.session_state.editor_key += 1
                st.rerun()
    
    # Right column - Output & Results
    with col2:
        st.header("Output Grid")
        
        # Run Pipeline Button
        run_disabled = not (health and health.get("status") == "healthy")
        
        if st.button(
            "Run Pipeline", 
            use_container_width=True, 
            type="primary",
            disabled=run_disabled
        ):
            with st.spinner("Running ML pipeline..."):
                try:
                    result = run_pipeline(
                        st.session_state.grid,
                        st.session_state.room_id
                    )
                    st.session_state.result = result
                    st.success("Pipeline completed successfully!")
                except requests.exceptions.RequestException as e:
                    st.error(f"Pipeline failed: {str(e)}")
        
        # Display results
        if st.session_state.result:
            result = st.session_state.result
            
            # Output grid visualization
            st.markdown(
                render_grid_html(result["output_matrix"], ""),
                unsafe_allow_html=True
            )
            
            # Metrics
            st.divider()
            st.subheader("Pipeline Results")
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    "Execution Time",
                    f"{result['execution_time_ms']:.1f} ms"
                )
            
            with metric_col2:
                st.metric(
                    "Stages Executed",
                    result['stages_executed']
                )
            
            with metric_col3:
                total_placements = sum(
                    s.get('placements', 0) 
                    for s in result['stage_summary'].values()
                )
                st.metric("Total Placements", total_placements)
            
            # Stage details
            st.divider()
            st.subheader("Stage Details")
            
            for stage_name, stage_data in result['stage_summary'].items():
                with st.expander(f"**{stage_name}**", expanded=True):
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.write(f"**Model:** {stage_data['model_selected']}")
                        st.write(f"**Version:** {stage_data['model_version']}")
                        st.write(f"**Placements:** {stage_data['placements']}")
                    
                    with detail_col2:
                        st.write(f"**Confidence:** {stage_data['confidence']:.2%}")
                        st.write(f"**Time:** {stage_data['execution_time_ms']:.2f} ms")
                        status = "Valid" if stage_data['is_valid'] else "Invalid"
                        st.write(f"**Status:** {status}")
            
            # Raw JSON (collapsible)
            with st.expander("Raw JSON Response"):
                st.json(result)
        else:
            st.info("Click 'Run Pipeline' to generate component placements")
    
    # Footer
    st.divider()
    st.caption(
        "Ceiling ML Pipeline UI | "
        "FastAPI Backend on port 8000 | "
        "Built with Streamlit"
    )


if __name__ == "__main__":
    main()
