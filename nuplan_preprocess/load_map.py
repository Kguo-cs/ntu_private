import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import fiona

def load_road_lines(gpkg_path: str):
    # Layer names are fixed by the nuPlan map schema.
    gdf = gpd.read_file(gpkg_path, layer='boundaries')   # columns include: geometry (LineString), type, ...
    print(fiona.listlayers(gpkg))

    print("CRS:", gdf.crs)  # ensure your coordinates are in the same CRS as your scenario data
    return gdf

def get_road_lines_in_radius(gdf: gpd.GeoDataFrame, x: float, y: float, r: float, types=None):
    """
    x, y in map (metric) coordinates; r in meters.
    types: iterable of string filters, e.g. {"ROAD_LINE_SOLID_SINGLE_WHITE", "ROAD_LINE_SOLID_DOUBLE_YELLOW"}
    Returns a GeoDataFrame subset.
    """
    # Spatial index for speed
    if gdf.sindex is None:
        _ = gdf.sindex

    # Quick bbox prefilter then precise filter
    query_geom = Point(x, y).buffer(r)
    cand = gdf.iloc[list(gdf.sindex.intersection(query_geom.bounds))]
    roi = cand[cand.intersects(query_geom)]

    if types is not None:
        roi = roi[roi["type"].isin(types)]
    return roi.reset_index(drop=True)

def linestrings_to_xy_arrays(roi: gpd.GeoDataFrame):
    """Convert LineString / MultiLineString to list of Nx2 numpy arrays."""
    import numpy as np
    xy_list = []
    for geom in roi.geometry:
        if geom.geom_type == "LineString":
            xy_list.append(np.asarray(geom.coords, dtype=float))
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                xy_list.append(np.asarray(part.coords, dtype=float))
    return xy_list

# --- Example ---
# pick your city map, e.g., Las Vegas Strip; adapt path to your machine
gpkg = "/home/ke/code/catk/nuplan_data/dataset/maps/us-nv-las-vegas-strip/9.15.1915/map.gpkg"

road_lines = load_road_lines(gpkg)
# select a region around (x, y) with 80 m radius, and keep only solid whites
roi = get_road_lines_in_radius(
    road_lines, x=1000.0, y=500.0, r=80.0,
    types={"ROAD_LINE_SOLID_SINGLE_WHITE", "ROAD_LINE_SOLID_DOUBLE_WHITE"}
)
xy_segments = linestrings_to_xy_arrays(roi)

print("Found segments:", len(xy_segments))
print("Unique types:", set(roi["type"]))
