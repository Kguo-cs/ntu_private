import numpy as np
import matplotlib.pyplot as plt


def interpolate_line(pts, num_points=100):
    """Interpolate a line (3,N) to have num_points evenly spaced along length."""
    # x, y, z = line["xyz"]
    # pts = np.stack([x, y, z], axis=1)  # (N,3)

    # Compute cumulative arc-length
    dist = np.cumsum(np.r_[0, np.linalg.norm(np.diff(pts, axis=0), axis=1)])
    total_len = dist[-1]
    target_dist = np.linspace(0, total_len, num_points)

    # Interpolate for each dimension
    new_pts = np.stack([np.interp(target_dist, dist, pts[:, d]) for d in range(3)], axis=1)
    return new_pts  # (num_points, 3)

def centerline(lane1, lane2, num_points=100):
    pts1 = interpolate_line(lane1, num_points)
    pts2 = interpolate_line(lane2, num_points)
    center = (pts1 + pts2) / 2.0
    return center  # (num_points, 3)

def decode_map_features_from_json(annotation,remove_mapid=[]):
    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    polylines = []
    # other_id=[]
    point_cnt = 0

    map_features=annotation['lines']+annotation["traffic_elements"]
    line_dict={}

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

            line_dict[id]=xyz
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


    for i,group in enumerate(annotation['lane_line_groups']):

        lane1=line_dict[group['lane_line_ids'][0]]
        lane2=line_dict[group['lane_line_ids'][1]]

        xyz=centerline(lane1, lane2)
        cur_info = {"id": 900+i}

        cur_info["type"] = 0

        cur_polyline = np.concatenate([xyz,np.zeros([len(xyz),1])+cur_info["type"],np.zeros([len(xyz),1])+cur_info["id"]],axis=-1)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

        map_infos["lane"].append(cur_info)

        # plt.plot(lane1[:,0],lane1[:,1],color='r')
        # plt.plot(lane2[:,0],lane2[:,1],color='b')
        # plt.show()
        #
        # print(1)

    map_infos["all_polylines_list"] = polylines

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")
    map_infos["all_polylines"] = polylines
    return map_infos