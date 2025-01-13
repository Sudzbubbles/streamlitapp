import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Generate a grid with values strictly greater than 0 and up to 1
def generate_grid(rows, cols, high_density_prob, low_density_prob, scale, generation_scale):
    if scale == "Parsec":
        resolution = 1
    elif scale == "Kiloparsec":
        resolution = 10

    rows, cols = rows // resolution, cols // resolution

    # Normalize probabilities to ensure valid distribution
    total_prob = high_density_prob + low_density_prob
    normalized_high = high_density_prob / total_prob
    normalized_low = low_density_prob / total_prob

    # Create the grid
    energy_values = np.linspace(0.01, 1, 10)
    probabilities = [normalized_low] * 5 + [normalized_high] * 5
    probabilities = np.array(probabilities) / np.sum(probabilities)

    grid = np.random.choice(energy_values, size=(rows, cols), p=probabilities)
    # Apply generation scale as contrast adjustment
    grid = grid ** generation_scale
    return grid

# Compute a secondary field based on the energy grid
def compute_secondary_field(energy_grid):
    return 1 / (1 + energy_grid)  # Example inverse relationship

# Initialize default scale in session_state
if "scale" not in st.session_state:
    st.session_state.scale = "Parsec"

# Streamlit UI
st.title("Pixel Pattern Emergence Grid")
st.sidebar.header("Controls")

# Grid Settings at the top
st.sidebar.subheader("Grid Settings")
grid_size = st.sidebar.slider("Grid Size", 10, 100, 50, step=10)

# Scale buttons under Grid Settings
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Parsec"):
        st.session_state.scale = "Parsec"
with col2:
    if st.button("Kiloparsec"):
        st.session_state.scale = "Kiloparsec"

# Energy Dispersion Parameters
st.sidebar.subheader("Regional Energy Dispersion")
low_density_prob = st.sidebar.slider("Low Dispersion Probability (%)", 1, 100, 50)
high_density_prob = st.sidebar.slider("High Dispersion Probability (%)", 1, 100, 50)

st.sidebar.subheader("Generation Scale")
generation_scale = st.sidebar.slider(
    "Generation Scale", 
    min_value=0.1, 
    max_value=2.0, 
    value=1.0, 
    step=0.1, 
    help="Adjusts the contrast between high and low dispersion regions, affecting pattern emergence."
)

# Toggle for zoom functionality
enable_zoom = st.sidebar.checkbox("Enable Zoom", value=True)

# Retain grid pattern across toggles
@st.cache_data
def get_energy_grid(grid_size, high_density_prob, low_density_prob, scale, generation_scale):
    return generate_grid(grid_size, grid_size, high_density_prob, low_density_prob, scale, generation_scale)

# Generate energy grid and compute secondary field
energy_grid = get_energy_grid(grid_size, high_density_prob, low_density_prob, st.session_state.scale, generation_scale)
secondary_field = compute_secondary_field(energy_grid)

# Add a toggle for the view
view_type = st.sidebar.radio(
    "View Grid Type", 
    ["Energy Dispersion", "Secondary Field"],
    help="Switch between Energy Dispersion and Secondary Field views."
)

# Prepare the data for the selected grid
if view_type == "Energy Dispersion":
    data = energy_grid
    color_scale = "Viridis"
    colorbar_title = "Energy Dispersion (0.01 to 1)"
else:
    data = secondary_field
    color_scale = "Plasma_r"  # Inverted colour scale for Secondary Field
    colorbar_title = "Secondary Field (Derived)"

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
        hoverinfo="z",  # Display the value of the patch directly
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

# Purpose Section (Collapsible at the bottom of the sidebar)
with st.sidebar.expander("Purpose"):
    st.markdown(
        """
        This app serves as a **Pixel Pattern Emergence Grid**, a sandbox for exploring how regional energy dispersion drives emergent behaviours in a probability-based system.
        
        - Adjust **probability parameters** to control the likelihood of high or low dispersion regions.
        - Modify the **generation scale** to amplify or smooth patterns.
        - Switch between views to explore different aspects of the simulated grid.

        This tool provides an experimental environment for understanding the role of regional probabilities in forming larger-scale patterns, offering a conceptual proof of emergent behaviours at various scales.
        """
    )

# Grid Views Explained Section (Collapsible at the bottom of the sidebar)
with st.sidebar.expander("Grid Views Explained"):
    st.markdown(
        """
        **Energy Dispersion View**:
        - **Lighter Colours**: Represent regions with higher energy dispersion.
        - **Darker Colours**: Represent regions with lower energy dispersion.

        **Secondary Field View**:
        - **Lighter Colours**: Represent derived values from higher dispersion regions.
        - **Darker Colours**: Represent derived values from lower dispersion regions.
        """
    )
