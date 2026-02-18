import numpy as np

N_x = 150
N_z = 60

rho_rock = 1.0
rho_void = 0.0
rho_ore = 2.5


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
        z = np.clip(z, 0, N_z - 1)

        # curvature makes tunnel drift horizontally
        x += curvature + rng.uniform(-0.5, 0.5)
        x = int(np.clip(x, 0, N_x - 1))

        path.append([int(x), int(z)])

        # carve out tunnel
        for dz in range(-thickness, thickness + 1):
            for dx in range(-thickness * 2, thickness * 2 + 1):
                zz = np.clip(z + dz, 0, N_z - 1)
                xx = np.clip(x + dx, 0, N_x - 1)
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
        cz = rng.randint(int(N_z * 0.2), N_z)
        w = rng.randint(5, 20)
        h = rng.randint(3, 10)
        grid[cz:cz + h, max(0, cx - w // 2):min(N_x, cx + w // 2)] = rho_ore

    # --- add void pockets (caves) ---
    for _ in range(3):
        cx = rng.randint(0, N_x)
        cz = rng.randint(0, N_z)
        w = rng.randint(8, 20)
        h = rng.randint(3, 7)
        grid[cz:cz + h, max(0, cx - w // 2):min(N_x, cx + w // 2)] = rho_void

    # --- add tunnels (0 to 3 tunnels) ---
    num_tunnels = rng.randint(0, 4)  # Changed from (1,3) to (0,4) to allow no tunnels
    tunnels_info = []

    for _ in range(num_tunnels):
        start_x = rng.randint(0, N_x)
        start_z = rng.randint(5, N_z // 2)
        length = rng.randint(40, 120)
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
