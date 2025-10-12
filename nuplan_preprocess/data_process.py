from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
import os
from  nuplan.common.actor_state.vehicle_parameters import  get_pacifica_parameters
from nuplan.planning.utils.multithreading.worker_parallel import  SingleMachineParallelExecutor
import numpy as np
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import RasterLayer, RasterMap, SemanticMapLayer
import pickle
from pathlib import Path
from tqdm import tqdm
import multiprocessing
# import ray
from multiprocessing import Pool

#gump_path='/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim' #'/home/ke/code/catk'#'/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim' # # #'/home/ke/code/catk'
gump_path=os.path.dirname(os.getcwd()) #'/home/ke/code/catk''/home/ke/keguo/sim'#
import sys

sys.path.append(gump_path)

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map
from src.data_preprocess import get_map_features,get_agent_features
import matplotlib as mpl
import torch

mpl.rcParams['toolbar'] = 'None'
scenario_mapping_config = {
    "scenario_map": {
        # scenario_name: [scenario_duration, extraction_offset]
        "accelerating_at_crosswalk": [15.0, -3.0],
        "accelerating_at_stop_sign": [15.0, -3.0],
        "accelerating_at_stop_sign_no_crosswalk": [15.0, -3.0],
        "accelerating_at_traffic_light": [15.0, -3.0],
        "accelerating_at_traffic_light_with_lead": [15.0, -3.0],
        "accelerating_at_traffic_light_without_lead": [15.0, -3.0],
        "behind_bike": [15.0, -3.0],
        "behind_long_vehicle": [15.0, -3.0],
        "behind_pedestrian_on_driveable": [15.0, -3.0],
        "behind_pedestrian_on_pickup_dropoff": [15.0, -3.0],
        "changing_lane": [15.0, -3.0],
        "changing_lane_to_left": [15.0, -3.0],
        "changing_lane_to_right": [15.0, -3.0],
        "changing_lane_with_lead": [15.0, -3.0],
        "changing_lane_with_trail": [15.0, -3.0],
        "crossed_by_bike": [15.0, -3.0],
        "crossed_by_vehicle": [15.0, -3.0],
        "following_lane_with_lead": [15.0, -3.0],
        "following_lane_with_slow_lead": [15.0, -3.0],
        "following_lane_without_lead": [15.0, -3.0],
        "high_lateral_acceleration": [15.0, -3.0],
        "high_magnitude_jerk": [15.0, -3.0],
        "high_magnitude_speed": [15.0, -3.0],
        "low_magnitude_speed": [15.0, -3.0],
        "medium_magnitude_speed": [15.0, -3.0],
        "near_barrier_on_driveable": [15.0, -3.0],
        "near_construction_zone_sign": [15.0, -3.0],
        "near_high_speed_vehicle": [15.0, -3.0],
        "near_long_vehicle": [15.0, -3.0],
        "near_multiple_bikes": [15.0, -3.0],
        "near_multiple_pedestrians": [15.0, -3.0],
        "near_multiple_vehicles": [15.0, -3.0],
        "near_pedestrian_at_pickup_dropoff": [15.0, -3.0],
        "near_pedestrian_on_crosswalk": [15.0, -3.0],
        "near_pedestrian_on_crosswalk_with_ego": [15.0, -3.0],
        "near_trafficcone_on_driveable": [15.0, -3.0],
        "on_all_way_stop_intersection": [15.0, -3.0],
        "on_carpark": [15.0, -3.0],
        "on_intersection": [15.0, -3.0],
        "on_pickup_dropoff": [15.0, -3.0],
        "on_stopline_crosswalk": [15.0, -3.0],
        "on_stopline_stop_sign": [15.0, -3.0],
        "on_stopline_traffic_light": [15.0, -3.0],
        "on_traffic_light_intersection": [15.0, -3.0],
        "starting_high_speed_turn": [15.0, -3.0],
        "starting_left_turn": [15.0, -3.0],
        "starting_low_speed_turn": [15.0, -3.0],
        "starting_protected_cross_turn": [15.0, -3.0],
        "starting_protected_noncross_turn": [15.0, -3.0],
        "starting_right_turn": [15.0, -3.0],
        "starting_straight_stop_sign_intersection_traversal": [15.0, -3.0],
        "starting_straight_traffic_light_intersection_traversal": [15.0, -3.0],
        "starting_u_turn": [15.0, -3.0],
        "starting_unprotected_cross_turn": [15.0, -3.0],
        "starting_unprotected_noncross_turn": [15.0, -3.0],
        "stationary": [15.0, -3.0],
        "stationary_at_crosswalk": [15.0, -3.0],
        "stationary_at_traffic_light_with_lead": [15.0, -3.0],
        "stationary_at_traffic_light_without_lead": [15.0, -3.0],
        "stationary_in_traffic": [15.0, -3.0],
        "stopping_at_crosswalk": [15.0, -3.0],
        "stopping_at_stop_sign_no_crosswalk": [15.0, -3.0],
        "stopping_at_stop_sign_with_lead": [15.0, -3.0],
        "stopping_at_stop_sign_without_lead": [15.0, -3.0],
        "stopping_at_traffic_light_with_lead": [15.0, -3.0],
        "stopping_at_traffic_light_without_lead": [15.0, -3.0],
        "stopping_with_lead": [15.0, -3.0],
        "traversing_crosswalk": [15.0, -3.0],
        "traversing_intersection": [15.0, -3.0],
        "traversing_narrow_lane": [15.0, -3.0],
        "traversing_pickup_dropoff": [15.0, -3.0],
        "traversing_traffic_light_intersection": [15.0, -3.0],
        "waiting_for_pedestrian_to_cross": [15.0, -3.0],
    }
}

# Example usage:
scenario_mapping = ScenarioMapping(subsample_ratio_override=0.5,**scenario_mapping_config)


os.environ["NUPLAN_DEVKIT_PATH"] =gump_path+ "/nuplan-devkit"
os.environ["NUPLAN_DATA_ROOT"] = gump_path+"/nuplan_data/dataset/nuplan-v1.1/splits/train"
os.environ["NUPLAN_MAPS_ROOT"] =gump_path+ "/nuplan_data/dataset/maps"
os.environ["NUPLAN_EXP_ROOT"] = gump_path

scenario_builder=NuPlanScenarioBuilder(
        data_root=os.getenv("NUPLAN_DATA_ROOT"),
        map_root=os.getenv("NUPLAN_MAPS_ROOT"),
        map_version='nuplan-maps-v1.0',
        sensor_root=os.getenv("NUPLAN_DATA_ROOT")+'/sensor_blobs',
        verbose= False,
        db_files=None,
        scenario_mapping=scenario_mapping,
        vehicle_parameters=get_pacifica_parameters()
)
worker=SingleMachineParallelExecutor(use_process_pool=False,max_workers=32)
scenario_filter=ScenarioFilter( scenario_types=None,
                                scenario_tokens=None,
                                log_names=None,
                                map_names=None,
                                limit_total_scenarios=None,
                                ego_displacement_minimum_m=None,
                                num_scenarios_per_type=20000,
                                timestamp_threshold_s=10,
                                remove_invalid_goals=False,
                                shuffle=False,
                                expand_scenarios=False)



#scenarios= scenario_builder.get_scenarios(scenario_filter, worker)

past_time_horizon=1
past_num_steps=10
future_time_horizon=8
future_num_steps=80
num_step = future_num_steps + past_num_steps + 1
output_dir = os.getenv("NUPLAN_EXP_ROOT") + '/src/waymo_data/full/nuplan_cross2'
scene_dir = os.getenv("NUPLAN_EXP_ROOT") + '/src/waymo_data/full'

os.makedirs(output_dir,exist_ok=True)
output_dir = Path(output_dir)
#print(len(scenarios))

# with open(Path(scene_dir) / f"scenarios.pkl", "wb+") as f:
#     pickle.dump(scenarios, f)
#
# print('finish scenarios filter')#373222
# print(1/0)

with open(Path(scene_dir) / f"scenarios.pkl", "rb+") as f:
    scenarios = pickle.load(f)

import matplotlib.pyplot as plt
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
import geopandas as gpd
from shapely.ops import unary_union
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from shapely.geometry import Point

def get_points_from_boundary(boundary):
    path = boundary.discrete_path
    points = [(pose.x, pose.y) for pose in path]
   # points = nuplan_to_metadrive_vector(points, center)
    return points

def boundaries_in_range(gdf: gpd.GeoDataFrame, cx: float, cy: float, radius_m: float,
                        types=None) -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame of boundary segments within radius_m of (cx, cy).
    Optionally filter by boundary_type_fid (types can be a set/list of ints/strings).
    """
    # Ensure geometry and sindex exist
    assert gdf.geometry.name == "geometry", "GeoDataFrame must have a 'geometry' column."
    _ = gdf.sindex  # build spatial index if not already

    ball = Point(cx, cy).buffer(radius_m)
    idxs = list(gdf.sindex.intersection(ball.bounds))
    cand = gdf.iloc[idxs]
    roi = cand[cand.intersects(ball)].copy()

    if types is not None and "boundary_type_fid" in roi.columns:
        roi = roi[roi["boundary_type_fid"].isin(types)].copy()

    return roi.reset_index(drop=True)

def extract_map_features(map_api, center,  radius):
    ret = {}
    #np.seterr(all='ignore')
    # Center is Important !
    layer_names = [
        #SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.LANE,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.INTERSECTION,
        #SemanticMapLayer.STOP_LINE,
        #SemanticMapLayer.WALKWAYS,
        SemanticMapLayer.CARPARK_AREA,
        SemanticMapLayer.ROADBLOCK,
       #SemanticMapLayer.ROADBLOCK_CONNECTOR,

        # unsupported yet
        # SemanticMapLayer.STOP_SIGN,
        # SemanticMapLayer.DRIVABLE_AREA,
    ]
    center_for_query = Point2D(*center)
    nearest_vector_map = map_api.get_proximal_map_objects(center_for_query, radius, layer_names)
    # boundaries = map_api._get_vector_map_layer(SemanticMapLayer.BOUNDARIES)
    #
    # inrange=boundaries_in_range(boundaries, center[0],center[1], radius)
    #
    # inrange.plot()

    # drivable_area =map_api._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA)
    # drivable_boundary=boundaries_in_range(drivable_area, center[0],center[1], radius)#.buffer(0.1).boundary.explode(index_parts=True)



    #drivable_boundary.plot()
    # boundary = drivable_boundary.union_all().boundary
    # if isinstance(boundary, LineString):
    #     block_points = np.array(boundary.xy).T
    #     #block_points = nuplan_to_metadrive_vector(block_points, center)
    #     id = "boundary_0"
    #     #ret[id] =block_points
    #     ret[id] = ('boundary', block_points)
    #
    # #     plt.plot(x, y, color='r')
    # #
    # elif isinstance(boundary, MultiLineString):
    #     for idx,line in enumerate(boundary.geoms):
    #         id = "boundary_{}".format(idx)
    #         block_points = np.array(line.xy).T
    #
    #         ret[id] =  ('boundary', block_points)

            # x, y = line.xy
    #         plt.plot(x, y, color='r')

    #plt.show()

    #drivable_boundary.plot()
   # plt.show()
    #
    # plt.savefig(Path(output_dir, f"map_{center}.png"))
    #
    # print(1/0)

    # Filter out stop polygons in turn stop
    # if SemanticMapLayer.STOP_LINE in nearest_vector_map:
    #     stop_polygons = nearest_vector_map[SemanticMapLayer.STOP_LINE]
    #     nearest_vector_map[SemanticMapLayer.STOP_LINE] = [
    #         stop_polygon for stop_polygon in stop_polygons if stop_polygon.stop_line_type != StopLineType.TURN_STOP
    #     ]
    #block_polygons = []

    # for lane in nearest_vector_map[SemanticMapLayer.LANE_CONNECTOR]:
    #     path = lane.baseline_path.discrete_path[::10]
    #     points = np.array([[pose.x, pose.y] for pose in path])
    #     ret[lane.id]=('lane',points)

    for lane in nearest_vector_map[SemanticMapLayer.LANE]:
        # path = lane.baseline_path.discrete_path
        # points = np.array([[pose.x, pose.y] for pose in path])
        # ret[lane.id]=('lane',points)

        left_nei,right_nei=lane.adjacent_edges
        left_boundary=lane.left_boundary
        right_boundary=lane.right_boundary

        if left_boundary.id not in ret.keys():
            if left_nei is not None:
            # if left_boundary.id in ret.keys():
            #     print(ret[left_boundary.id][0]=='broken')
            # else:
                ret[left_boundary.id]=('broken',get_points_from_boundary(left_boundary))


            # if (left_nei.id, lane.id) not in couple:
            #     couple.append((left_nei.id, lane.id))
            #     # broken_dict[(left_nei.id, lane.id)] = lane.left_boundary
            #     ret['broken_'+left_nei.id+'_' + lane.id] =get_points_from_boundary(lane.left_boundary, center)
            # else:
            #     print(left_nei.id, lane.id)
            else:
                #solid_dict[lane.id] = lane.left_boundary
                # ret[left.id] = {'solid_line': get_points_from_boundary(left, center)}
                # ret['solid_left_'+lane.id] = get_points_from_boundary(lane.left_boundary, center)
                # if left_boundary.id in ret.keys():
                #     print(ret[left_boundary.id][0]=='solid')
                # else:
                ret[left_boundary.id]=('solid',get_points_from_boundary(left_boundary))

        if right_boundary.id not in ret.keys():
            if right_nei is not None:
            # if right_boundary.id in ret.keys():
            #     print(ret[right_boundary.id][0]=='broken')
            # else:
                ret[right_boundary.id]=('broken',get_points_from_boundary(right_boundary))

            # if (lane.id,right_nei.id) not in couple:
            #     couple.append((lane.id,right_nei.id))
            #     ret['broken_'+lane.id+'_' + right_nei.id] = get_points_from_boundary(lane.right_boundary, center)

                # broken_dict[(lane.id,right_nei.id)] = lane.right_boundary
            # else:
            #     print(lane.id,right_nei.id)
            else:
            # if right_boundary.id in ret.keys():
            #     print(ret[right_boundary.id][0]=='solid')
            # else:
                ret[right_boundary.id]=('solid',get_points_from_boundary(right_boundary))

            # solid_dict[lane.id] = lane.right_boundary
            # ret['solid_right_'+lane.id] = get_points_from_boundary(lane.right_boundary, center)

    #print(1)

    # for layer in [SemanticMapLayer.ROADBLOCK]:
    #     for block in nearest_vector_map[layer]:
            # edges = sorted(block.interior_edges, key=lambda lane: lane.index) \
            #     if layer == SemanticMapLayer.ROADBLOCK else block.interior_edges
            # for index, lane_meta_data in enumerate(edges):
            #     if not hasattr(lane_meta_data, "baseline_path"):
            #         continue
            #     if isinstance(lane_meta_data.polygon.boundary, MultiLineString):
            #         boundary = gpd.GeoSeries(lane_meta_data.polygon.boundary).explode(index_parts=True)
            #         sizes = []
            #         for idx, polygon in enumerate(boundary[0]):
            #             sizes.append(len(polygon.xy[1]))
            #         points = boundary[0][np.argmax(sizes)].xy
            #     elif isinstance(lane_meta_data.polygon.boundary, LineString):
            #         points = lane_meta_data.polygon.boundary.xy
            #     polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
            #     # polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])
            #
            #     # According to the map attributes, lanes are numbered left to right with smaller indices being on the
            #     # left and larger indices being on the right.
            #     # @ See NuPlanLane.adjacent_edges()
            #     # ret[lane_meta_data.id] = {
            #     #     SD.TYPE: MetaDriveType.LANE_SURFACE_STREET \
            #     #         if layer == SemanticMapLayer.ROADBLOCK else MetaDriveType.LANE_SURFACE_UNSTRUCTURE,
            #     #     SD.POLYLINE: extract_centerline(lane_meta_data, center),
            #     #     SD.ENTRY: [edge.id for edge in lane_meta_data.incoming_edges],
            #     #     SD.EXIT: [edge.id for edge in lane_meta_data.outgoing_edges],
            #     #     SD.LEFT_NEIGHBORS: [edge.id for edge in block.interior_edges[:index]] \
            #     #         if layer == SemanticMapLayer.ROADBLOCK else [],
            #     #     SD.RIGHT_NEIGHBORS: [edge.id for edge in block.interior_edges[index + 1:]] \
            #     #         if layer == SemanticMapLayer.ROADBLOCK else [],
            #     #     SD.POLYGON: polygon,
            #     #     "is_sdc_route": lane_meta_data.get_roadblock_id() in route_block_ids,
            #     #     "speed_limit_mps": lane_meta_data.speed_limit_mps,
            #     # }
            #
            #     left_neighbors =  [edge.id for edge in block.interior_edges[:index]]  if layer == SemanticMapLayer.ROADBLOCK else []
            #
            #     ret[lane_meta_data.id]={'lane':polygon}
            #     if layer == SemanticMapLayer.ROADBLOCK_CONNECTOR:
            #         continue
            #     left = lane_meta_data.left_boundary
            #     if len(left_neighbors)>0:
            #         # only broken line in nuPlan data
            #         # line_type = get_line_type(int(boundaries.loc[[str(left.id)]]["boundary_type_fid"]))
            #         # line_type = MetaDriveType.LINE_BROKEN_SINGLE_WHITE
            #         #if line_type != MetaDriveType.LINE_UNKNOWN:
            #     # if len(left_neighbors)!=0:
            #         #print(len(left_neighbors))
            #         ret[left.id] = {'broken_line': get_points_from_boundary(left, center)}
            #     else:
            #        #print(len(left_neighbors))
            #        ret[left.id] = {'solid_line': get_points_from_boundary(left, center)}



            # if layer == SemanticMapLayer.ROADBLOCK:
            #     block_polygons.append(block.polygon)

    # walkway
    # for area in nearest_vector_map[SemanticMapLayer.WALKWAYS]:
    #     if isinstance(area.polygon.exterior, MultiLineString):
    #         boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
    #         sizes = []
    #         for idx, polygon in enumerate(boundary[0]):
    #             sizes.append(len(polygon.xy[1]))
    #         points = boundary[0][np.argmax(sizes)].xy
    #     elif isinstance(area.polygon.exterior, LineString):
    #         points = area.polygon.exterior.xy
    #     polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
    #     #polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])
    #     ret[area.id] = {
    #         'sidewalk': polygon,
    #         # SD.POLYGON: polygon,
    #     }
    #
    # corsswalk
    for area in nearest_vector_map[SemanticMapLayer.CROSSWALK]:
        if isinstance(area.polygon.exterior, MultiLineString):
            boundary = gpd.GeoSeries(area.polygon.exterior).explode(index_parts=True)
            sizes = []
            for idx, polygon in enumerate(boundary[0]):
                sizes.append(len(polygon.xy[1]))
            points = boundary[0][np.argmax(sizes)].xy
        elif isinstance(area.polygon.exterior, LineString):
            points = area.polygon.exterior.xy
        polygon = [[points[0][i], points[1][i]] for i in range(len(points[0]))]
        # polygon = nuplan_to_metadrive_vector(polygon, nuplan_center=[center[0], center[1]])
        ret[area.id] = ('cross_walk', polygon)


    block_polygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.ROADBLOCK]]
    carpark_polygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.CARPARK_AREA]]
    interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
    boundaries = gpd.GeoSeries(unary_union(block_polygons+interpolygons+carpark_polygons)).boundary.explode(index_parts=True)
    # interpolygons = [block.polygon for block in nearest_vector_map[SemanticMapLayer.INTERSECTION]]
    # boundaries = gpd.GeoSeries(unary_union(interpolygons))#.boundary.explode(index_parts=True)
   # boundaries.plot()

   # plt.show()
    # plt.show()
    for idx, boundary in enumerate(boundaries[0]):
        block_points = np.array(list(i for i in zip(boundary.coords.xy[0], boundary.coords.xy[1])))
        #block_points = nuplan_to_metadrive_vector(block_points, center)
        id = "boundary_{}".format(idx)
        #ret[id] =block_points
        ret[id] = ('boundary', block_points)
    return ret


def get_map_vector(scenario,origin_ego,center,radius):#373222

    result = extract_map_features(scenario.map_api, center, radius)

    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}

    polylines = []
    point_cnt = 0

    for id,(key,line) in enumerate(result.values()):

        if 'solid' in key:
           # plt.plot(np.array(line)[:,0],np.array(line)[:,1],'red')

            line_type=7
        elif 'cross_walk' in key:
            #plt.plot(np.array(line)[:,0],np.array(line)[:,1],'green')
            line_type=9
        elif 'broken' in key:
           # plt.plot(np.array(line)[:,0],np.array(line)[:,1],'blue')

            line_type=6
        elif 'boundary' in key:
            #plt.plot(np.array(line)[:,0],np.array(line)[:,1],'cyan',alpha=0.5)
            line_type=4
        elif 'lane' in key:
            #plt.plot(np.array(line)[:,0],np.array(line)[:,1],'grey')
            line_type=1
            continue


        cur_info = {"id": id,"type":line_type}

        xyz=np.array(line)

        cur_polyline = np.concatenate(
            [xyz,np.zeros([len(xyz), 1]), np.zeros([len(xyz), 1]) + line_type, np.zeros([len(xyz), 1]) +id], axis=-1)

        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

        if line_type == 7 or line_type == 6:
            map_infos["road_line"].append(cur_info)
        elif line_type == 4:
            map_infos["road_edge"].append(cur_info)
        elif line_type == 9:
            map_infos["crosswalk"].append(cur_info)
        elif line_type == 1:
            map_infos["lane"].append(cur_info)

  #  plt.show()

    try:
        polylines=np.concatenate(polylines, axis=0)

        polylines[:,:2]-=origin_ego[None]

        polylines = polylines.astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")

    map_infos["all_polylines"] = polylines

    # signal_state = {
    #     0: "LANE_STATE_GO",
    #     #  States for traffic signals with arrows.
    #     1: "LANE_STATE_CAUTION",
    #     2: "LANE_STATE_STOP",
    #     3: "LANE_STATE_UNKNOWN",
    # }

    # tf_current_light = scenario.get_traffic_light_status_at_iteration(0)
    #
    # dynamic_map_infos = {"lane_id": [], "state": []}
    # lane_id, state = [], []
    # for cur_signal in tf_current_light:  # (num_observed_signals)
    #     lane_id.append(cur_signal.lane_connector_id)
    #     state.append(signal_state[cur_signal.status])
    #
    # dynamic_map_infos["lane_id"].append(np.array([lane_id]))
    # dynamic_map_infos["state"].append(np.array([state]))
    #
    # tf_lights = process_dynamic_map(dynamic_map_infos)
    # tf_current_light = tf_lights.loc[tf_lights["time_step"] == 0]

    map_data = get_map_features(map_infos, {})

    data = preprocess_map(map_data,break_dist=300)



    return data


from shapely.prepared import prep

import numpy as np
import matplotlib.pyplot as plt
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
from shapely.prepared import prep
from matplotlib.patches import Polygon as MplPolygon

# --- 2) Helper: add shapely polygon(s) to matplotlib axes ---
def add_poly_or_multipoly(ax, geom, *, facecolor=None, edgecolor="k", alpha=1.0, lw=0.8):
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        exterior = np.asarray(geom.exterior.coords, float)
        patch = MplPolygon(exterior, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=lw)
        ax.add_patch(patch)
        # holes (optional): draw with no facecolor
        for ring in geom.interiors:
            ring_xy = np.asarray(ring.coords, float)
            ax.add_patch(MplPolygon(ring_xy, closed=True, facecolor="none", edgecolor=edgecolor, alpha=alpha*0.6, linewidth=lw*0.6))
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            add_poly_or_multipoly(ax, g, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, lw=lw)

def get_agent(scenario,origin_ego):

    detections = scenario.get_tracked_objects_at_iteration(past_num_steps)

    id_mapping = {}
    idx = 1
    drivable_area =scenario.map_api._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA)
    drivable_range=boundaries_in_range(drivable_area, origin_ego[0],origin_ego[1], 300)#.buffer(0.1).boundary.explode(index_parts=True)

    for agent in detections.tracked_objects:
        track_type = agent.tracked_object_type.value
        if track_type < 3:
            track_token = agent.track_token
            id_mapping[track_token] = idx
            idx += 1
        else:
            # static_geoms.append(agent.box.geometry)
            if  drivable_range.intersects(agent.box.geometry).any():
                track_token = agent.track_token
                id_mapping[track_token] = idx
                idx += 1
                #static_geoms.append(agent.box.geometry)


        # VEHICLE = 0, 'vehicle'
        # PEDESTRIAN = 1, 'pedestrian'
        # BICYCLE = 2, 'bicycle'
        # TRAFFIC_CONE = 3, 'traffic_cone'
        # BARRIER = 4, 'barrier'
        # CZONE_SIGN = 5, 'czone_sign'
        # GENERIC_OBJECT = 6, 'generic_object'
        # EGO = 7, 'ego'

    # drivable_area =scenario.map_api._get_vector_map_layer(SemanticMapLayer.DRIVABLE_AREA)
    # drivable_boundary=boundaries_in_range(drivable_area, origin_ego[0],origin_ego[1], 500)#.buffer(0.1).boundary.explode(index_parts=True)
    #
    #
    # fig, ax = plt.subplots(figsize=(8, 8))
    #
    # # Drivable area (filled light gray)
    # add_poly_or_multipoly(ax, drivable_union, facecolor="#d0d0d0", edgecolor="#888", alpha=0.8, lw=0.6)
    #
    # # Static objects (red outlines, semi-opaque fill)
    # for g in static_geoms:
    #     add_poly_or_multipoly(ax, g, facecolor="#ffdddd", edgecolor="r", alpha=0.8, lw=1.2)
    #
    # # Nice view
    # ax.set_aspect("equal", adjustable="box")
    # ax.set_title("Drivable area (gray) with static objects (red)")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    #
    # # Optional: center/zoom around ego origin and 500 m radius
    # cx, cy = origin_ego
    # rad = 500.0
    # ax.set_xlim(cx - rad, cx + rad)
    # ax.set_ylim(cy - rad, cy + rad)
    #
    # plt.tight_layout()
    # plt.show()

    num_agent = idx

    track_infos = {"role": np.zeros([num_agent, 3], dtype=bool),
                   "object_id": np.arange(num_agent),
                   "valid": np.zeros([num_agent, num_step], dtype=bool),
                   "object_type": np.zeros(num_agent, dtype=np.uint8),
                   "states": np.zeros([num_agent, num_step, 9], dtype=np.float32)
                   }

    track_infos["role"][0][0] = True

    for t in range(num_step):
        detections = scenario.get_tracked_objects_at_iteration( t)
        ego_state = scenario.get_ego_state_at_iteration(t)

        agent = ego_state.agent

        track_infos["valid"][0][t] = True
        track_infos["states"][0][t][0] = agent.center.x-origin_ego[0]
        track_infos["states"][0][t][1] = agent.center.y-origin_ego[1]
        track_infos["states"][0][t][3] = agent.box.length
        track_infos["states"][0][t][4] = agent.box.width
        track_infos["states"][0][t][5] = agent.box.height
        track_infos["states"][0][t][6] = agent.center.heading
        track_infos["states"][0][t][7] = agent.velocity.x
        track_infos["states"][0][t][8] = agent.velocity.y

        for agent in detections.tracked_objects:
            track_token = agent.track_token
            track_type = agent.tracked_object_type.value

            if track_token in id_mapping.keys() :
                track_idx = id_mapping[track_token]
                track_infos["valid"][track_idx][t] = True
                track_infos["object_type"][track_idx] = track_type

                track_infos["states"][track_idx][t][0] = agent.center.x-origin_ego[0]
                track_infos["states"][track_idx][t][1] = agent.center.y-origin_ego[1]
                track_infos["states"][track_idx][t][3] = agent.box.length
                track_infos["states"][track_idx][t][4] = agent.box.width
                track_infos["states"][track_idx][t][5] = agent.box.height
                track_infos["states"][track_idx][t][6] = agent.center.heading
                track_infos["states"][track_idx][t][7] = agent.velocity.x
                track_infos["states"][track_idx][t][8] = agent.velocity.y

    out_dict = get_agent_features(track_infos,'train', past_num_steps + 1,num_step )

    # position=out_dict["position"]
    # valid=out_dict["valid_mask"]
    #
    # valid_pos=position[valid][:,:2]+origin_ego[None]
    #
    # plt.scatter(valid_pos[:,0], valid_pos[:,1])
    #
    #plt.show()
    
    return out_dict

def process_scenario(scenario):

    ego_state = scenario.get_ego_state_at_iteration(10)

    origin_ego=np.array([ego_state.center.x,ego_state.center.y])


    agent=get_agent(scenario,origin_ego)

    position=agent["position"]
    valid=agent["valid_mask"]

    valid_pos=position[valid]

    valid_pos=valid_pos.numpy()
    valid_pos_max=np.max(valid_pos,axis=0)
    valid_pos_min=np.min(valid_pos,axis=0)

    center_pos=(valid_pos_max+valid_pos_min)/2

    gap=np.linalg.norm(valid_pos-center_pos[None],axis=1).max()

    center_pos=center_pos[:2]+origin_ego

    # plt.scatter(valid_pos[:,0], valid_pos[:,1])
    #
    # plt.show()

    #print(gap)

    data=get_map_vector(scenario,origin_ego,center_pos,gap+60)

    data["agent"]=agent


    #filter map
    # map_pos=data['map_save']['traj_pos'][:,0]
    #
    # agent_pos=agent['position'][:,:,:2].reshape(-1,2)
    #
    # dist=torch.norm(agent_pos[:,None]-map_pos[None],dim=-1)
    #
    # min_dist=dist.amin(0)
    #
    # mask=min_dist<60
    #
    # data['map_save']['traj_pos']=data['map_save']['traj_pos'][mask]
    # data['map_save']['traj_theta']=data['map_save']['traj_theta'][mask]
    # data['pt_token']['type']=data['pt_token']['type'][mask]
    # data['pt_token']['pl_type']=data['pt_token']['pl_type'][mask]
    # data['pt_token']['num_nodes']=len(data['pt_token']['pl_type'])

    del data['pt_token']['light_type']
    # traj_pos = data['map_save']['traj_pos']
    #
    # # type=data['pt_token']['type']
    #
    # # print(len(traj_pos))
    # #
    # # print(len(traj_pos[type==6]))
    # #
    # for traj in traj_pos:  # [type==6]
    #
    #     plt.plot(traj[:, 0], traj[:, 1], '-')
    # plt.show()

    scenario_id=scenario.token

    with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
        pickle.dump(data, f)


# with multiprocessing.Pool(28) as p:
#     r = list(tqdm(p.imap_unordered(process_scenario, scenarios), total=len(scenarios)))
#print(len(scenarios))

#
# with Pool(28) as pool:
#     results = pool.starmap(process_scenario, zip(scenarios))
with Pool(32) as pool:
    results = list(tqdm(pool.imap_unordered(process_scenario, scenarios), total=len(scenarios)))
# for scenario in tqdm(scenarios):
#     process_scenario(scenario)

# # Submit tasks in parallel
# futures = [process_scenario.remote(scenario) for scenario in scenarios]
#
# # Optional: use tqdm to show progress
# for _ in tqdm(ray.get(futures), desc="Processing scenarios"):
#     pass
#
# ray.shutdown()
