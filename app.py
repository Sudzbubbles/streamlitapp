import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Generate a density grid
def generate_grid(rows, cols, high_density_prob, low_density_prob):
    grid = np.random.choice(
        [0.1, 1],  # Low density (0.1), High density (1)
        size=(rows, cols),
        p=[low_density_prob / 100, high_density_prob / 100],
    )
    return grid

# Compute timeflow from density
def compute_timeflow(density_grid):
    return 1 / density_grid

# Streamlit UI
st.title("Interactive Map of Regional Timeflow")
st.sidebar.header("Controls")

# Grid size slider
grid_size = st.sidebar.slider("Grid Size", 10, 100, 50, step=10)

# Probability sliders
high_density_prob = st.sidebar.slider("High-Density Probability (H%)", 0, 100, 50)
low_density_prob = st.sidebar.slider("Low-Density Probability (L%)", 0, 100, 50)

# Generate a density grid with normalized probabilities
def generate_grid(rows, cols, high_density_prob, low_density_prob):
    # Ensure the probabilities sum to 1
    total_prob = high_density_prob + low_density_prob
    if total_prob == 0:
        high_density_prob = 50
        low_density_prob = 50
        total_prob = 100

    normalized_high = high_density_prob / total_prob
    normalized_low = low_density_prob / total_prob

    # Generate the grid
    grid = np.random.choice(
        [0.1, 1],  # Low density (0.1), High density (1)
        size=(rows, cols),
        p=[normalized_low, normalized_high],
    )
    return grid

# Plot the grid
fig, ax = plt.subplots(figsize=(8, 8))
norm = Normalize(vmin=timeflow_grid.min(), vmax=timeflow_grid.max())
im = ax.imshow(timeflow_grid, cmap="viridis", norm=norm)
plt.colorbar(im, ax=ax, label="Timeflow")

# Annotate cells for small grids
if grid_size <= 20:
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, f"{density_grid[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white")

ax.set_title("Regional Timeflow Map")
ax.set_xlabel("Regions (X-axis)")
ax.set_ylabel("Regions (Y-axis)")
st.pyplot(fig)
