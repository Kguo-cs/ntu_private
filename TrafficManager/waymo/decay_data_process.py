import numpy as np


def decode_map_features_from_json(map_features,remove_mapid=[]):
    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    polylines = []
    # other_id=[]
    point_cnt = 0

    map_features=map_features['lines']+map_features["traffic_elements"]

    for mf in map_features:
        id=mf['global_id']

        if id in remove_mapid:
            continue

        feature_data_type=mf['class']
        xyz=np.array(mf['xyz']).T
        cur_info = {"id": id}

        if feature_data_type=="lane_line":
            line_type = mf['attrs']["laneline_type"]
            if line_type=="solid":
                cur_info["type"] = 7
            else:
                cur_info["type"] = 6
        elif feature_data_type=="boundary":
            cur_info["type"] = 4
        elif feature_data_type=="speed_bump" or feature_data_type=="crosswalk":
            cur_info["type"] = 9
        # elif feature_data_type=="arrow":
        #     continue
        else:
            continue

        cur_polyline = np.concatenate([xyz,np.zeros([len(xyz),1])+cur_info["type"],np.zeros([len(xyz),1])+cur_info["id"]],axis=-1)

        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

        if feature_data_type=="lane_line":
            map_infos["road_line"].append(cur_info)
        elif feature_data_type == "boundary":
            map_infos["road_edge"].append(cur_info)
        elif feature_data_type == "speed_bump":
            map_infos["crosswalk"].append(cur_info)

    map_infos["all_polylines_list"] = polylines

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")
    map_infos["all_polylines"] = polylines
    return map_infos