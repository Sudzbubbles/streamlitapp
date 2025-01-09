import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Generate a density grid with values strictly greater than 0 and up to 1
def generate_grid(rows, cols, high_density_prob, low_density_prob):
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

# Functionality for the "Purpose" section
def show_purpose():
    st.markdown(
        """
        ### **Purpose**
        This interactive app demonstrates a **timescape cosmology model**, showing the relationship between **density** and **timeflow** in a structured grid.
        
        It demonstrates that **timeflow is inversely proportional to density, and timeflow changes inversely with density**:
        - **High-density regions (clusters)** result in **slower timeflow**.
        - **Low-density regions (voids)** lead to **faster timeflow**.

        By visualising these dynamics, the app provides an intuitive understanding of how local density variations affect time dilation and regional timeflow, serving both scientific exploration and educational purposes.
        """
    )

# Functionality for the "Explain Grid Views" section
def show_grid_views():
    st.markdown(
        """
        ### **Grid Views Explained**
        **Density Grid**:
        - **Lighter Colours**: Represent **higher density regions (clusters)**.
        - **Darker Colours**: Represent **lower density regions (voids)**.
        
        **Timeflow Grid**:
        - **Lighter Colours**: Represent **slower timeflow**, corresponding to **higher density regions (clusters)**.
        - **Darker Colours**: Represent **faster timeflow**, corresponding to **lower density regions (voids)**.
        """
    )

# Add buttons in the sidebar
if st.sidebar.button("Purpose"):
    show_purpose()

if st.sidebar.button("Explain Grid Views"):
    show_grid_views()

# Grid size slider
grid_size = st.sidebar.slider("Grid Size", 10, 100, 50, step=10)

# Probability sliders
low_density_prob = st.sidebar.slider("Low-Density Probability (L%)", 0, 100, 50)
high_density_prob = st.sidebar.slider("High-Density Probability (H%)", 0, 100, 50)

# Retain grid pattern across toggles
@st.cache_data
def get_density_grid(grid_size, high_density_prob, low_density_prob):
    return generate_grid(grid_size, grid_size, high_density_prob, low_density_prob)

# Generate density grid (cached) and compute timeflow grid
density_grid = get_density_grid(grid_size, high_density_prob, low_density_prob)
timeflow_grid = compute_timeflow(density_grid)

# Add a toggle for the view
view_type = st.sidebar.radio(
    "View Grid Type", 
    ["Density", "Timeflow"],
    help="Switch between density and timeflow views."
)

# Add a note below the controls to clarify the relationship
if view_type == "Timeflow":
    st.sidebar.markdown(
        """
        **Timeflow Grid View**:
        - **Lighter Colours**: Represent **slower timeflow**, corresponding to **higher density regions (clusters)**.
        - **Darker Colours**: Represent **faster timeflow**, corresponding to **lower density regions (voids)**.
        """
    )
else:
    st.sidebar.markdown(
        """
        **Density Grid View**:
        - **Lighter Colours**: Represent **higher density regions (clusters)**.
        - **Darker Colours**: Represent **lower density regions (voids)**.
        """
    )

# Plot the selected grid
fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)

if view_type == "Density":
    # Plot the density grid
    norm = Normalize(vmin=0.01, vmax=1)  # Updated range for density
    im = ax.imshow(density_grid, cmap="viridis", norm=norm)
    plt.colorbar(im, ax=ax, label="Density (0.01 to 1)")
    ax.set_title("Density Grid")
else:
    # Plot the timeflow grid with reversed colormap
    norm = Normalize(vmin=timeflow_grid.min(), vmax=timeflow_grid.max())  # Dynamic timeflow range
    im = ax.imshow(timeflow_grid, cmap="plasma_r", norm=norm)  # Reversed colormap
    plt.colorbar(im, ax=ax, label="Timeflow (Slow to Fast)")
    ax.set_title("Timeflow Grid (Slow = High Density, Fast = Low Density)")

# Display the plot
ax.set_xlabel("Regions (X-axis)")
ax.set_ylabel("Regions (Y-axis)")
st.pyplot(fig)
