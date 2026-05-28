import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata


def generate_perlin_landscape(
    width=256, height=256, scale=50, octaves=4, persistence=0.5, lacunarity=2.0
):
    """Generate a Perlin-noise-like landscape using layered interpolation."""
    noise = np.zeros((height, width))
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        grid_size = int(scale / frequency)
        grid_x = np.linspace(0, width, grid_size)
        grid_y = np.linspace(0, height, grid_size)

        random_values = np.random.rand(len(grid_y), len(grid_x))

        x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
        grid_points = np.column_stack([x_indices.ravel(), y_indices.ravel()])
        sample_points = np.column_stack(
            [np.tile(grid_x, len(grid_y)), np.repeat(grid_y, len(grid_x))]
        )

        octave_noise = griddata(
            sample_points, random_values.ravel(), grid_points, method="cubic"
        )
        octave_noise = octave_noise.reshape((height, width))
        octave_noise = gaussian_filter(octave_noise, sigma=frequency)

        noise += octave_noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    noise = noise / max_amplitude
    noise = np.clip(noise, 0, 1)
    return noise


print("Generating large-clump Perlin noise landscape...")

# CHANGED PARAMETERS:
# 1. Increased persistence to 1.8 so large-scale, blurred octaves dominate.
# 2. Kept scale=40 to prevent the internal grid_size from dropping below 2 (which crashes griddata).
landscape = generate_perlin_landscape(
    width=80, height=80, scale=40, octaves=5, persistence=1.8, lacunarity=2.0
)

# This percentile approach automatically locks the layout to exactly 80% territory coverage
threshold = np.percentile(landscape, 20)
terrain_map = (landscape > threshold).astype(np.uint8)

output_path = "maps/random.npy"
np.save(output_path, terrain_map ^ 1)
print(f"Map saved to {output_path}")
print(f"Map shape: {terrain_map.shape}")
print(f"Passable terrain: {100 * terrain_map.mean():.1f}%")
