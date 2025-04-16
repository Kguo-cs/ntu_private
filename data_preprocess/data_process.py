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
gump_path='/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim' #'/home/ke/code/GUMP' # #'/home/ke/code/GUMP' #'/home/users/ntu/lyuchen/scratch/keguo_projects/gump' ##~/scratch/keguo_projects/gump'
import sys

sys.path.append('/home/users/ntu/lyuchen/scratch/keguo_projects/ntu/sim')

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from data_preprocess.process import get_polylines_from_polygon, preprocess_map,get_map_features,process_dynamic_map,get_agent_features

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


os.environ["NUPLAN_DEVKIT_PATH"] =gump_path+ "/third_party/nuplan-devkit"
os.environ["NUPLAN_DATA_ROOT"] = gump_path+"/nuplan_data/dataset/nuplan-v1.1/splits/train"
os.environ["NUPLAN_MAPS_ROOT"] =gump_path+ "/nuplan_data/dataset/maps"
os.environ["NUPLAN_EXP_ROOT"] = gump_path+'/exp'

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
worker=SingleMachineParallelExecutor(use_process_pool=False,max_workers=1)
scenario_filter=ScenarioFilter(
                                scenario_types=None,
                                scenario_tokens=None,
                                log_names=None,
                                map_names=None,
                                limit_total_scenarios=None,
                                ego_displacement_minimum_m=None,
                               num_scenarios_per_type=10,
                               timestamp_threshold_s=10,
                               remove_invalid_goals=True,
                               shuffle=False,
                               expand_scenarios=False)



scenarios= scenario_builder.get_scenarios(scenario_filter, worker)


def get_map_vector(scenario):
    ego_state = scenario.get_ego_state_at_iteration(0)

    origin = Point2D(ego_state.center.x, ego_state.center.y)

    map_api = scenario.map_api
    map_infos = {"lane": [], "crosswalk": []}

    lanes = map_api.get_proximal_map_objects(origin, radius=200,
                                             layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR,
                                                     SemanticMapLayer.CROSSWALK
                                                     ])

    polylines = []
    point_cnt = 0

    for lane in lanes[SemanticMapLayer.LANE]:
        baseline = np.array(lane.baseline_path.linestring.coords.xy)
        id = int(lane.id)
        cur_info = {"id": id, "type": 0}

        cur_polyline = np.stack(
            [baseline[0], baseline[1], np.zeros([len(baseline[0])]), id + np.zeros([len(baseline[0])])], axis=-1)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        map_infos["lane"].append(cur_info)
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

        # left_boundary=lane.left_boundary.linestring.coords.xy
        #
        # cur_polyline = np.stack( [left_boundary,1+np.zeros([len(left_boundary),1]),len(polylines)+np.zeros([len(left_boundary),1])],axis=-1 )
        # polylines.append(cur_polyline)#RoadLine
        #
        # right_boundary=lane.right_boundary.linestring.coords.xy
        # cur_polyline = np.stack( [right_boundary,1+np.zeros([len(right_boundary),1]),len(polylines)+np.zeros([len(right_boundary),1])],axis=-1 )
        # polylines.append(cur_polyline)#RoadLine

    for cross_walk in lanes[SemanticMapLayer.CROSSWALK]:
        xy = np.array(cross_walk.polygon.boundary.coords)
        xyz = np.concatenate([xy, np.zeros([len(xy), 1])], axis=-1)
        polygon_idx = np.linspace(0, xyz.shape[0], 4, endpoint=False, dtype=int)
        pl_polygon = get_polylines_from_polygon(xyz[polygon_idx])
        id = int(cross_walk.id)

        cur_info = {"id": id, "type": 1}

        cur_polyline = np.stack(
            [pl_polygon[0], pl_polygon[1], np.zeros([len(pl_polygon[0])]), id + np.zeros([len(pl_polygon[0])])],
            axis=-1)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        map_infos["crosswalk"].append(cur_info)
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

    for lane_connector in lanes[SemanticMapLayer.LANE_CONNECTOR]:
        baseline = lane_connector.baseline_path.linestring.coords.xy
        id = int(lane_connector.id)
        cur_info = {"id": id, "type": 0}

        cur_polyline = np.stack(
            [baseline[0], baseline[1], np.zeros([len(baseline[0])]), id + np.zeros([len(baseline[0])])], axis=-1)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        map_infos["lane"].append(cur_info)
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")

    map_infos["all_polylines"] = polylines

    signal_state = {
        0: "LANE_STATE_GO",
        #  States for traffic signals with arrows.
        1: "LANE_STATE_CAUTION",
        2: "LANE_STATE_STOP",
        3: "LANE_STATE_UNKNOWN",
    }

    tf_current_light = scenario.get_traffic_light_status_at_iteration(0)

    dynamic_map_infos = {"lane_id": [], "state": []}
    lane_id, state = [], []
    for cur_signal in tf_current_light:  # (num_observed_signals)
        lane_id.append(cur_signal.lane_connector_id)
        state.append(signal_state[cur_signal.status])

    dynamic_map_infos["lane_id"].append(np.array([lane_id]))
    dynamic_map_infos["state"].append(np.array([state]))

    tf_lights = process_dynamic_map(dynamic_map_infos)
    tf_current_light = tf_lights.loc[tf_lights["time_step"] == 0]

    map_data = get_map_features(map_infos, tf_current_light)

    data = preprocess_map(map_data)

    return data

past_time_horizon=1
past_num_steps=10
future_time_horizon=8
future_num_steps=80
num_step = future_num_steps + past_num_steps + 1


def get_agent(scenario):

    detections = scenario.get_tracked_objects_at_iteration(past_num_steps)

    id_mapping = {}
    idx = 1

    for agent in detections.tracked_objects:
        track_type = agent.tracked_object_type.value
        if track_type < 3:
            track_token = agent.track_token
            id_mapping[track_token] = idx
            idx += 1

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
        track_infos["states"][0][t][0] = agent.center.x
        track_infos["states"][0][t][1] = agent.center.y
        track_infos["states"][0][t][3] = agent.box.length
        track_infos["states"][0][t][4] = agent.box.width
        track_infos["states"][0][t][5] = agent.box.height
        track_infos["states"][0][t][6] = agent.center.heading
        track_infos["states"][0][t][7] = agent.velocity.x
        track_infos["states"][0][t][8] = agent.velocity.y

        for agent in detections.tracked_objects:
            track_token = agent.track_token
            track_type = agent.tracked_object_type.value

            if track_token in id_mapping.keys() and track_type < 3:
                track_idx = id_mapping[track_token]
                track_infos["valid"][track_idx][t] = True
                track_infos["object_type"][track_idx] = track_type

                track_infos["states"][track_idx][t][0] = agent.center.x
                track_infos["states"][track_idx][t][1] = agent.center.y
                track_infos["states"][track_idx][t][3] = agent.box.length
                track_infos["states"][track_idx][t][4] = agent.box.width
                track_infos["states"][track_idx][t][5] = agent.box.height
                track_infos["states"][track_idx][t][6] = agent.center.heading
                track_infos["states"][track_idx][t][7] = agent.velocity.x
                track_infos["states"][track_idx][t][8] = agent.velocity.y

    out_dict = get_agent_features(track_infos, past_num_steps + 1,num_step )
    
    return out_dict

print(len(scenarios))
for scenario in tqdm(scenarios):

    data=get_map_vector(scenario)
    scenario_id=scenario.token
    output_dir = os.getenv("NUPLAN_EXP_ROOT") + '/src/waymo_data/full/nuplan'
    output_dir = Path(output_dir)
    data["agent"]=get_agent(scenario)

    with open(output_dir / f"{scenario_id}.pkl", "wb+") as f:
        pickle.dump(data, f)
