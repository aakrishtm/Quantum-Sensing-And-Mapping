import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "training_data"

# Load a single grid
grid = np.load("training_data/density_grid_000.npy")

plt.figure(figsize=(7, 3))
plt.imshow(grid, origin='upper', cmap='inferno')
plt.colorbar(label="density")
plt.xlabel("x index")
plt.ylabel("z index (depth)")
plt.title("Density Grid")
plt.show()

# Load all grids (discover count from directory)
grid_filenames = [f for f in os.listdir(DATA_DIR) if f.startswith("density_grid_") and f.endswith(".npy")]
grids = [np.load(os.path.join(DATA_DIR, f)) for f in sorted(grid_filenames)]