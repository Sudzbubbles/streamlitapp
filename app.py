import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Generate a density grid with values strictly greater than 0 and up to 1
def generate_grid(rows, cols, high_density_prob, low_density_prob, scale, generation_scale):
    if scale == "Parsec":
        resolution = 1
    elif scale == "Kiloparsec":
        resolution = 10

    rows, cols = rows // resolution, cols // resolution

    total_prob = high_density_prob + low_density_prob
    if total_prob == 0:
        high_density_prob, low_density_prob = 50, 50
        total_prob = 100

    normalized_high = high_density_prob / total_prob
    normalized_low = low_density_prob / total_prob

    # Create the grid
    density_values = np.linspace(0.01, 1, 10)
    probabilities = [normalized_low] * 5 + [normalized_high] * 5
    probabilities = np.array(probabilities) / np.sum(probabilities)

    grid = np.random.choice(density_values, size=(rows, cols), p=probabilities)
    # Apply generation scale as contrast adjustment
    grid = grid ** generation_scale
    return grid

# Compute timeflow with a smooth gradient
def compute_timeflow(density_grid):
    return 1 / (1 + density_grid)

# Precompute hover region medians for each patch
def precompute_hover_medians(data, region_size):
    medians = np.zeros_like(data)
    half_size = region_size // 2

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            y_min = max(0, i - half_size)
            y_max = min(data.shape[0], i + half_size + 1)
            x_min = max(0, j - half_size)
            x_max = min(data.shape[1], j + half_size + 1)
            hover_region = data[y_min:y_max, x_min:x_max]
            medians[i, j] = np.median(hover_region)

    return medians

# Streamlit UI
st.title("Interactive Cosmological Map of Regional Timeflow and Density")
st.sidebar.header("Controls")

# Set default scale in session state
if "scale" not in st.session_state:
    st.session_state.scale = "Parsec"

# Set default hover mode in session state
if "hover_mode" not in st.session_state:
    st.session_state.hover_mode = "Single Patch"

# Grid size slider
grid_size = st.sidebar.slider("Grid Size", 10, 100, 50, step=10)

# Scale buttons under Grid Size
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Parsec"):
        st.session_state.scale = "Parsec"
with col2:
    if st.button("Kiloparsec"):
        st.session_state.scale = "Kiloparsec"

# Probability sliders
low_density_prob = st.sidebar.slider("Low-Density Probability (L%)", 0, 100, 50)
high_density_prob = st.sidebar.slider("High-Density Probability (H%)", 0, 100, 50)

# Generation Scale slider
generation_scale = st.sidebar.slider(
    "Generation Scale", 
    min_value=0.1, 
    max_value=2.0, 
    value=1.0, 
    step=0.1, 
    help="Adjusts the contrast between high-density (clusters) and low-density (voids), affecting the timeflow variations."
)

# Hover mode buttons
st.sidebar.subheader("Hover Display")
if st.sidebar.button("Single Patch Value"):
    st.session_state.hover_mode = "Single Patch"
if st.sidebar.button("2x2 Median"):
    st.session_state.hover_mode = "2x2 Median"
if st.sidebar.button("3x3 Median"):
    st.session_state.hover_mode = "3x3 Median"

# Toggle for zoom functionality
enable_zoom = st.sidebar.checkbox("Enable Zoom", value=True)

# Retain grid pattern across toggles
@st.cache_data
def get_density_grid(grid_size, high_density_prob, low_density_prob, scale, generation_scale):
    return generate_grid(grid_size, grid_size, high_density_prob, low_density_prob, scale, generation_scale)

# Generate density grid and compute timeflow grid based on selected scale and generation scale
density_grid = get_density_grid(grid_size, high_density_prob, low_density_prob, st.session_state.scale, generation_scale)
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
    color_scale = "Plasma_r"  # Inverted colour scale for Timeflow
    colorbar_title = "Timeflow (Fast to Slow)"

# Compute hover values based on selected hover mode
if st.session_state.hover_mode == "Single Patch":
    hover_values = data
    hover_template = "<b>Value: %{z:.2f}</b><extra></extra>"
elif st.session_state.hover_mode == "2x2 Median":
    hover_values = precompute_hover_medians(data, 2)
    hover_template = "<b>Median (2x2): %{customdata:.2f}</b><extra></extra>"
elif st.session_state.hover_mode == "3x3 Median":
    hover_values = precompute_hover_medians(data, 3)
    hover_template = "<b>Median (3x3): %{customdata:.2f}</b><extra></extra>"

# Create an interactive heatmap using Plotly
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=data,
        colorscale=color_scale,
        colorbar=dict(
            title=colorbar_title,
            title_side="right",  # Vertically aligned
            title_font=dict(size=18),  # Larger font for better readability
        ),
        customdata=hover_values,  # Custom data for hover
        hovertemplate=hover_template,  # Dynamically updated template
    )
)

# Configure layout for better UI scaling
fig.update_layout(
    width=800,
    height=800,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(scaleanchor="y", constrain="domain"),
    yaxis=dict(scaleanchor="x", constrain="domain"),
    dragmode="pan" if enable_zoom else False,
)

# Display the figure
st.plotly_chart(fig, use_container_width=True)
