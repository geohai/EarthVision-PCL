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
    lat_offset=0.0,   # NEW
    lon_offset=0.0,   # NEW
):
    """
    Plot a checkerboard grid and color data points by fold.

    lat_offset / lon_offset shift the checkerboard origin (in degrees), consistent
    with checkerboard_deg_fold_indices:
        row = floor(((lat - min_lat) - lat_offset) / grid_deg)
        col = floor(((lon - min_lon) - lon_offset) / grid_deg)
    """
    coords = np.asarray(coords, dtype=float)
    lat = coords[:, 0]
    lon = coords[:, 1]

    scale_lower = str(scale).lower()

    # Determine bounding box (match splits.py behavior as closely as possible)
    states = None
    if scale_lower == "global":
        min_lat, max_lat = -90.0, 90.0
        min_lon, max_lon = -180.0, 180.0
    elif scale_lower == "conus":
        states = gpd.read_file(geojson_url)
        if "id" in states.columns:
            states = states[~states["id"].isin(exclude_state_ids)]
        states = states.to_crs(epsg=4326)
        min_lon, min_lat, max_lon, max_lat = states.total_bounds  # x,y order
    else:
        min_lat, max_lat = np.nanmin(lat), np.nanmax(lat)
        min_lon, max_lon = np.nanmin(lon), np.nanmax(lon)

    # --- Build grid polygons with fold IDs (support negative row/col due to offsets) ---
    # Compute the row/col index range needed to cover the bbox, given the shifted origin.
    # We want all cells whose [x1,x2] and [y1,y2] cover [min_lon,max_lon] and [min_lat,max_lat].
    row_start = int(np.floor((0.0 - float(lat_offset)) / float(grid_deg)))
    row_end   = int(np.ceil(((max_lat - min_lat) - float(lat_offset)) / float(grid_deg)))
    col_start = int(np.floor((0.0 - float(lon_offset)) / float(grid_deg)))
    col_end   = int(np.ceil(((max_lon - min_lon) - float(lon_offset)) / float(grid_deg)))

    grid_polys, grid_fold_ids = [], []
    for col_idx in range(col_start, col_end):
        for row_idx in range(row_start, row_end):
            x1 = min_lon + float(lon_offset) + col_idx * float(grid_deg)
            y1 = min_lat + float(lat_offset) + row_idx * float(grid_deg)
            poly = box(x1, y1, x1 + float(grid_deg), y1 + float(grid_deg))
            grid_polys.append(poly)
            grid_fold_ids.append((row_idx + col_idx) % int(n_splits))

    grid = gpd.GeoDataFrame({"geometry": grid_polys, "fold": grid_fold_ids}, crs="EPSG:4326")

    # --- Assign folds to points using the SAME formula as splits.py ---
    valid = (~np.isnan(lat)) & (~np.isnan(lon))
    fold_ids = np.full(len(lat), -1, dtype=int)
    if valid.any():
        row = np.floor(((lat[valid] - min_lat) - float(lat_offset)) / float(grid_deg)).astype(int)
        col = np.floor(((lon[valid] - min_lon) - float(lon_offset)) / float(grid_deg)).astype(int)
        fold_ids[valid] = (row + col) % int(n_splits)

    points = gpd.GeoDataFrame(
        {"fold": fold_ids},
        geometry=[Point(lon_i, lat_i) for lon_i, lat_i in zip(lon, lat)],
        crs="EPSG:4326",
    )

    # Reproject (EPSG:5070 is fine for CONUS; for global you may prefer a global projection,
    # but keeping as-is is okay if you're only using CONUS.)
    grid_proj = grid.to_crs("EPSG:5070")
    points_proj = points.to_crs("EPSG:5070")
    states_proj = states.to_crs("EPSG:5070") if states is not None else None

    cmap = ListedColormap(["blue", "red", "green", "orange", "purple", "brown"][:n_splits])
    norm = BoundaryNorm(np.arange(n_splits + 1) - 0.5, ncolors=n_splits)

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

    legend_patches = [mpatches.Patch(color=cmap(i), label=f"Partition {i}") for i in range(n_splits)]
    ax.legend(handles=legend_patches, title="Checkerboard Partitions", loc="upper right")
    ax.set_xticks([])
    ax.set_yticks([])

    title = f"Checkerboard Split (deg={grid_deg}, folds={n_splits}, lat_off={lat_offset}, lon_off={lon_offset})"
    ax.set_title(title, fontsize=14)

    # Optional: encode offsets in filename to avoid overwriting
    def _tag(x):
        s = f"{float(x):.3f}"
        return s.replace("-", "m").replace(".", "p")
    fig_path = os.path.join(
        run_dir,
        f"checkerboard_deg{grid_deg}_n{n_splits}_lat{_tag(lat_offset)}_lon{_tag(lon_offset)}.png",
    )

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved checkerboard visualization to {fig_path}")

