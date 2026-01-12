import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

def visualize_checkerboard_split(
    coords,
    grid_deg,
    n_splits,
    scale,
    run_dir,
    fold_index=0,
    geojson_url="https://github.com/mapbox/mapboxgl-jupyter/raw/refs/heads/master/examples/data/us-states.geojson",
    exclude_state_ids=("02","15","72"),
):
    """
    Plot a checkerboard grid and color data points by fold.

    Parameters
    ----------
    coords : ndarray of shape (N, 2)
        [lat, lon] for each sample.
    grid_deg : float
        Size of each grid cell in degrees (same as in config).
    n_splits : int
        Number of checkerboard partitions.
    scale : {'global', 'conus'}
        If 'conus', fetches the CONUS bounding box from GeoJSON.
        Otherwise uses dataset bounds.
    run_dir : str
        Directory to write the figure (e.g. run_results_dir).
    fold_index : int
        Index of the test fold; optional if you want to highlight test points.
    """
    coords = np.asarray(coords, dtype=float)
    lat = coords[:,0]
    lon = coords[:,1]

    # Determine bounding box
    if scale.lower() == "conus":
        # load states and drop AK/HI/PR
        states = gpd.read_file(geojson_url)
        if "id" in states.columns:
            states = states[~states["id"].isin(exclude_state_ids)]
        states = states.to_crs(epsg=4326)
        min_lon, min_lat, max_lon, max_lat = states.total_bounds  # x,y order
    else:
        min_lat, max_lat = np.nanmin(lat), np.nanmax(lat)
        min_lon, max_lon = np.nanmin(lon), np.nanmax(lon)

    # Build grid polygons with fold IDs
    cols = int(np.ceil((max_lon - min_lon) / float(grid_deg)))
    rows = int(np.ceil((max_lat - min_lat) / float(grid_deg)))
    grid_polys, grid_fold_ids = [], []
    for i in range(cols):
        for j in range(rows):
            x1 = min_lon + i * grid_deg
            y1 = min_lat + j * grid_deg
            poly = box(x1, y1, x1 + grid_deg, y1 + grid_deg)
            grid_polys.append(poly)
            grid_fold_ids.append((i + j) % n_splits)
    grid = gpd.GeoDataFrame({"geometry": grid_polys, "fold": grid_fold_ids}, crs="EPSG:4326")

    # Assign folds by computing row/col indices
    row = np.floor((lat - min_lat) / float(grid_deg)).astype(int)
    col = np.floor((lon - min_lon) / float(grid_deg)).astype(int)
    fold_ids = (row + col) % int(n_splits)

    # Create point GeoDataFrame
    points = gpd.GeoDataFrame(
        {"fold": fold_ids},
        geometry=[Point(lon_i, lat_i) for lon_i, lat_i in zip(lon, lat)],
        crs="EPSG:4326",
    )

    # Optionally reproject to a cartographic projection (e.g. NAD83 / Conus Albers)
    grid_proj = grid.to_crs("EPSG:5070")
    points_proj = points.to_crs("EPSG:5070")
    states_proj = states.to_crs("EPSG:5070") if scale.lower() == "conus" else None

    # Prepare colormap
    cmap = ListedColormap(["blue", "red", "green", "orange", "purple", "brown"][:n_splits])
    norm = BoundaryNorm(np.arange(n_splits + 1) - 0.5, ncolors=n_splits)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    if states_proj is not None:
        states_proj.plot(ax=ax, color="none", edgecolor="black", linewidth=1)
    grid_proj.plot(ax=ax, column="fold", cmap=cmap, alpha=0.5, edgecolor="black")
    ax.scatter(
        points_proj.geometry.x,
        points_proj.geometry.y,
        c=points_proj["fold"],
        cmap=cmap,
        norm=norm,
        s=5,
        alpha=0.6,
        zorder=5,
        edgecolors="k",
    )
    legend_patches = [
        mpatches.Patch(color=cmap(i), label=f"Partition {i}")
        for i in range(n_splits)
    ]
    ax.legend(handles=legend_patches, title="Checkerboard Partitions", loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"Checkerboard Split (deg={grid_deg}, folds={n_splits})"
    ax.set_title(title, fontsize=14)
    fig_path = os.path.join(run_dir, f"checkerboard_deg{grid_deg}_n{n_splits}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved checkerboard visualization to {fig_path}")
