import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Generate a density grid with a smooth distribution
def generate_grid(rows, cols, high_density_prob, low_density_prob):
    # Ensure probabilities sum to 1
    total_prob = high_density_prob + low_density_prob
    if total_prob == 0:
        high_density_prob = 50
        low_density_prob = 50
        total_prob = 100

    normalized_high = high_density_prob / total_prob
    normalized_low = low_density_prob / total_prob

    # Create a 10-step range of densities between 0.1 and 1
    density_values = np.linspace(0.1, 1, 10)

    # Create a matching probability distribution for the densities
    # First 5 values favor low density, last 5 favor high density
    probabilities = [normalized_low] * 5 + [normalized_high] * 5

    # Normalize the probabilities to ensure they sum to 1
    probabilities = np.array(probabilities) / np.sum(probabilities)

    # Generate the grid using the adjusted probabilities
    grid = np.random.choice(density_values, size=(rows, cols), p=probabilities)
    return grid

# Compute timeflow with a smooth gradient
def compute_timeflow(density_grid):
    # Timeflow is inversely proportional to density
    return np.exp(-density_grid)

# Streamlit UI
st.title("Interactive Map of Regional Timeflow and Density")
st.sidebar.header("Controls")

# Grid size slider
grid_size = st.sidebar.slider("Grid Size", 10, 100, 50, step=10)

# Probability sliders
high_density_prob = st.sidebar.slider("High-Density Probability (H%)", 0, 100, 50)
low_density_prob = st.sidebar.slider("Low-Density Probability (L%)", 0, 100, 50)

# Generate grid and compute timeflow
density_grid = generate_grid(grid_size, grid_size, high_density_prob, low_density_prob)
timeflow_grid = compute_timeflow(density_grid)

# Add a toggle for the view
view_type = st.sidebar.radio("View Grid Type", ["Timeflow", "Density"])

# Plot the selected grid
if view_type == "Density":
    # Plot the density grid
    fig, ax = plt.subplots(figsize=(8, 8))
    norm = Normalize(vmin=density_grid.min(), vmax=density_grid.max())
    im = ax.imshow(density_grid, cmap="viridis", norm=norm)
    plt.colorbar(im, ax=ax, label="Density")
    ax.set_title("Density Grid")
else:
    # Plot the timeflow grid
    fig, ax = plt.subplots(figsize=(8, 8))
    norm = Normalize(vmin=timeflow_grid.min(), vmax=timeflow_grid.max())
    im = ax.imshow(timeflow_grid, cmap="plasma", norm=norm)
    plt.colorbar(im, ax=ax, label="Timeflow")
    ax.set_title("Timeflow Grid")

# Display the plot
ax.set_xlabel("Regions (X-axis)")
ax.set_ylabel("Regions (Y-axis)")
st.pyplot(fig)
