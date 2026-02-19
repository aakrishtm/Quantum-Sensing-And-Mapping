import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving files
import matplotlib.pyplot as plt
import json

N_x = 150
N_z = 60

rho_rock = 1.0
rho_void = 0.0
rho_ore  = 2.5


def draw_tunnel(grid, tunnel_mask, start_x, start_z, length, thickness, curvature=0.0, rng=None):
    """
    Draw a tunnel as a continuous void path.
    curvature >0 bends right; <0 bends left.
    Returns tunnel metadata including path coordinates.
    """
    x, z = start_x, start_z
    path = []

    for step in range(length):
        # random walk in depth
        z += rng.randint(-1, 2)
        z = np.clip(z, 0, N_z-1)

        # curvature makes tunnel drift horizontally
        x += curvature + rng.uniform(-0.5, 0.5)
        x = int(np.clip(x, 0, N_x-1))

        path.append([int(x), int(z)])

        # carve out tunnel
        for dz in range(-thickness, thickness+1):
            for dx in range(-thickness*2, thickness*2+1):
                zz = np.clip(z+dz, 0, N_z-1)
                xx = np.clip(x+dx, 0, N_x-1)
                grid[zz, xx] = rho_void
                tunnel_mask[zz, xx] = 1  # Mark as tunnel in segmentation mask

    # Calculate bounding box from path
    path_array = np.array(path)
    bbox = [
        int(path_array[:, 0].min()),
        int(path_array[:, 1].min()),
        int(path_array[:, 0].max()),
        int(path_array[:, 1].max())
    ]

    return {
        "start_x": int(start_x),
        "start_z": int(start_z),
        "length": int(length),
        "thickness": int(thickness),
        "curvature": float(curvature),
        "path": path,
        "bounding_box": bbox
    }


def make_grid(seed):
    rng = np.random.RandomState(seed)
    grid = np.full((N_z, N_x), rho_rock)
    tunnel_mask = np.zeros((N_z, N_x), dtype=np.uint8)  # Binary mask: 0=no tunnel, 1=tunnel

    # --- add ore blobs ---
    for _ in range(5):
        cx = rng.randint(0, N_x)
        cz = rng.randint(int(N_z*0.2), N_z)
        w = rng.randint(5, 20)
        h = rng.randint(3, 10)
        grid[cz:cz+h, max(0, cx-w//2):min(N_x, cx+w//2)] = rho_ore

    # --- add void pockets (caves) ---
    for _ in range(3):
        cx = rng.randint(0, N_x)
        cz = rng.randint(0, N_z)
        w = rng.randint(8, 20)
        h = rng.randint(3, 7)
        grid[cz:cz+h, max(0, cx-w//2):min(N_x, cx+w//2)] = rho_void

    # --- add tunnels (0 to 3 tunnels) ---
    num_tunnels = rng.randint(0, 4)  # Changed from (1,3) to (0,4) to allow no tunnels
    tunnels_info = []
    
    for _ in range(num_tunnels):
        start_x = rng.randint(0, N_x)
        start_z = rng.randint(5, N_z//2)
        length   = rng.randint(40, 120)
        thickness = rng.randint(1, 3)
        curvature = rng.uniform(-0.3, 0.3)
        tunnel_info = draw_tunnel(grid, tunnel_mask, start_x, start_z, length, thickness, curvature, rng)
        tunnels_info.append(tunnel_info)

    # Create metadata
    metadata = {
        "seed": int(seed),
        "has_tunnel": bool(num_tunnels > 0),
        "num_tunnels": int(num_tunnels),
        "tunnels": tunnels_info,
        "grid_shape": [int(N_z), int(N_x)]
    }

    return grid, tunnel_mask, metadata


import os

# --- Generate and save 50 training maps ---
output_dir = "training_data"
os.makedirs(output_dir, exist_ok=True)

num_samples = 500

print(f"Generating {num_samples} density grids with labels...")

tunnel_count = 0
for i in range(num_samples):
    grid, tunnel_mask, metadata = make_grid(seed=i)
    
    # Save density grid as numpy array
    np.save(os.path.join(output_dir, f"density_grid_{i:03d}.npy"), grid)
    
    # Save tunnel mask (ground truth for segmentation)
    np.save(os.path.join(output_dir, f"tunnel_mask_{i:03d}.npy"), tunnel_mask)
    
    # Save metadata as JSON
    with open(os.path.join(output_dir, f"metadata_{i:03d}.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save visualization (density grid + mask overlay)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    
    # Density grid
    im1 = axes[0].imshow(grid, origin='upper', cmap='inferno')
    axes[0].set_title(f"Density Grid #{i:03d}")
    axes[0].set_xlabel("x index")
    axes[0].set_ylabel("z index (depth)")
    plt.colorbar(im1, ax=axes[0], label="density")
    
    # Tunnel mask (ground truth)
    im2 = axes[1].imshow(tunnel_mask, origin='upper', cmap='binary', vmin=0, vmax=1)
    has_tunnel_text = "YES" if metadata["has_tunnel"] else "NO"
    axes[1].set_title(f"Tunnel Mask (Has Tunnel: {has_tunnel_text})")
    axes[1].set_xlabel("x index")
    axes[1].set_ylabel("z index (depth)")
    plt.colorbar(im2, ax=axes[1], label="tunnel (0=no, 1=yes)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{i:03d}.png"), dpi=100)
    plt.close()
    
    if metadata["has_tunnel"]:
        tunnel_count += 1
    
    if (i + 1) % 10 == 0:
        print(f"  Generated {i + 1}/{num_samples}")

print(f"\nDone! Saved {num_samples} samples to '{output_dir}/'")
print(f"  - Samples with tunnels: {tunnel_count}/{num_samples} ({100*tunnel_count/num_samples:.1f}%)")
print(f"  - Samples without tunnels: {num_samples-tunnel_count}/{num_samples} ({100*(num_samples-tunnel_count)/num_samples:.1f}%)")
print(f"\nFiles saved per sample:")
print(f"  - density_grid_XXX.npy: input density grid ({N_z}x{N_x})")
print(f"  - tunnel_mask_XXX.npy: ground truth segmentation mask ({N_z}x{N_x})")
print(f"  - metadata_XXX.json: labels and tunnel info")
print(f"  - sample_XXX.png: visualization")
