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

    # Generate a continuous density grid
    grid = np.random.choice(
        np.linspace(0.1, 1, 10),  # Generate densities between 0.1 and 1
        size=(rows, cols),
        p=[normalized_low] * 5 + [normalized_high] * 5,  # Bias probabilities
    )
    return grid

# Compute timeflow with a smooth gradient
def compute_timeflow(density_grid):
    # Timeflow decays exponentially with density
    return np.exp(-density_grid)

# Streamlit UI
st.title("Interactive Map of Regional Timeflow")
st.sidebar.header("Controls")

# Grid size slider
grid_size = st.sidebar.slider("Grid Size", 10, 100, 50, step=10)

# Probability sliders
high_density_prob = st.sidebar.slider("High-Density Probability (H%)", 0, 100, 50)
low_density_prob = st.sidebar.slider("Low-Density Probability (L%)", 0, 100, 50)

# Generate grid and compute timeflow
density_grid = generate_grid(grid_size, grid_size, high_density_prob, low_density_prob)
timeflow_grid = compute_timeflow(density_grid)

# Plot the grid
fig, ax = plt.subplots(figsize=(8, 8))
norm = Normalize(vmin=timeflow_grid.min(), vmax=timeflow_grid.max())
im = ax.imshow(timeflow_grid, cmap="plasma", norm=norm)
plt.colorbar(im, ax=ax, label="Timeflow")

# Annotate cells for small grids
if grid_size <= 20:
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, f"{density_grid[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")

ax.set_title("Regional Timeflow Map")
ax.set_xlabel("Regions (X-axis)")
ax.set_ylabel("Regions (Y-axis)")
st.pyplot(fig)
