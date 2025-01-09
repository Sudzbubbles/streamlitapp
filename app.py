import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Generate a density grid with values strictly greater than 0 and up to 1
def generate_grid(rows, cols, high_density_prob, low_density_prob, scale):
    # Adjust grid resolution based on scale
    if scale == "Parsec":
        resolution = 1  # Finest resolution
    elif scale == "Kiloparsec":
        resolution = 10  # Coarser resolution

    rows, cols = rows // resolution, cols // resolution  # Adjust grid size

    # Ensure probabilities sum to 1
    total_prob = high_density_prob + low_density_prob
    if total_prob == 0:
        high_density_prob = 50
        low_density_prob = 50
        total_prob = 100

    normalized_high = high_density_prob / total_prob
    normalized_low = low_density_prob / total_prob

    # Create a range of densities strictly greater than 0 and up to 1
    density_values = np.linspace(0.01, 1, 10)

    # Create a matching probability distribution for the densities
    probabilities = [normalized_low] * 5 + [normalized_high] * 5
    probabilities = np.array(probabilities) / np.sum(probabilities)

    # Generate the grid using the adjusted probabilities
    grid = np.random.choice(density_values, size=(rows, cols), p=probabilities)
    return grid

# Compute timeflow with a smooth gradient
def compute_timeflow(density_grid):
    # Timeflow is inversely proportional to density
    return 1 / (1 + density_grid)

# Streamlit UI
st.title("Interactive Map of Regional Timeflow and Density")
st.sidebar.header("Controls")

# Scale selector: Parsec or Kiloparsec
scale = st.sidebar.selectbox("Select Scale", ["Parsec", "Kiloparsec"])

# Probability sliders
low_density_prob = st.sidebar.slider("Low-Density Probability (L%)", 0, 100, 50)
high_density_prob = st.sidebar.slider("High-Density Probability (H%)", 0, 100, 50)

# Toggle for zoom functionality
enable_zoom = st.sidebar.checkbox("Enable Zoom", value=True)

# Toggle for hover functionality
enable_hover = st.sidebar.checkbox("Enable Hover Median Display", value=True)

# Grid size slider
grid_size = st.sidebar.slider("Grid Size", 10, 100, 50, step=10)

# Retain grid pattern across toggles
@st.cache_data
def get_density_grid(grid_size, high_density_prob, low_density_prob, scale):
    return generate_grid(grid_size, grid_size, high_density_prob, low_density_prob, scale)

# Generate density grid and compute timeflow grid
density_grid = get_density_grid(grid_size, high_density_prob, low_density_prob, scale)
timeflow_grid = compute_timeflow(density_grid)

# Add a toggle for the view
view_type = st.sidebar.radio(
    "View Grid Type", 
    ["Density", "Timeflow"],
    help="Switch between density and timeflow views."
)

# Prepare the data for the selected grid
if view_type == "Density":
    data = density_grid
    color_scale = "Viridis"
    colorbar_title = "Density (0.01 to 1)"
else:
    data = timeflow_grid
    color_scale = "Plasma"
    colorbar_title = "Timeflow (Slow to Fast)"

# Create an interactive heatmap using Plotly
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=data,
        colorscale=color_scale,
        colorbar=dict(title=colorbar_title),
        hoverinfo="z" if enable_hover else "skip",
    )
)

# Configure layout for better UI scaling
fig.update_layout(
    width=800,  # Set a fixed width for the plot
    height=800,  # Set a fixed height for the plot
    margin=dict(l=10, r=10, t=10, b=10),  # Minimal margins for a clean look
    xaxis=dict(scaleanchor="y", constrain="domain"),  # Keep axes square
    yaxis=dict(scaleanchor="x", constrain="domain"),
    dragmode="pan" if enable_zoom else False,  # Enable/disable panning
)

# Add hover median functionality
if enable_hover:
    fig.update_traces(
        hovertemplate="<b>Value: %{z:.2f}</b><extra></extra>",
    )

# Display the figure
st.plotly_chart(fig, use_container_width=True)
