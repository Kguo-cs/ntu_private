import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from .desay_lane_graph import build_lane_graph_with_connectors
from .desay_edge_graph import build_edge_graph_from_lane_graph_topo
from .plot_lane_graph import  plot_lane_graph
from .plot_edge_graph import  plot_edge_graph

from .build_centerline import build_centerlines_from_boundaries_xyz

def decode_map_features_from_json(annotation,remove_mapid=[]):
    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    polylines = []
    # other_id=[]
    point_cnt = 0

    map_features=annotation['lines']+annotation["traffic_elements"]
    line_dict={}
    boundary_dict={}

    max_id=0

    for mf in map_features:
        id=mf['global_id']

        if id in remove_mapid:
            continue

        feature_data_type=mf['class']
        xyz=np.array(mf['xyz']).T
        cur_info = {"id": id}
        max_id=max(max_id,id)

        if feature_data_type=="lane_line":
            line_type = mf['attrs']["laneline_type"]
            if line_type=="solid":
                cur_info["type"] = 7
              #  plt.plot(xyz[:, 0], xyz[:, 1], color='r')
                #plt.plot(xyz[:2, 0], xyz[:2, 1], color='b')

            else:#dot
                cur_info["type"] = 6
                # print(line_type)
                #plt.plot(xyz[:, 0], xyz[:, 1], color='g')
                #plt.plot(xyz[:2, 0], xyz[:2, 1], color='b')
                line_dict[id]=xyz

        elif feature_data_type=="boundary":
            cur_info["type"] = 4
            #plt.plot(xyz[:, 0], xyz[:, 1], color='y')
            #plt.plot(xyz[:2, 0], xyz[:2, 1], color='b')
            #
            # if  id ==60:#id ==60 or
            boundary_dict[id]=xyz
                #plt.show()

               # print(id)#60,37

        elif feature_data_type == "speed_bump" or feature_data_type=="crosswalk":
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


    centerlines=build_centerlines_from_boundaries_xyz(boundary_dict)
    # ##line_dict,
    # for centerline in centerlines:
    #
    #     center=centerline.centerline
    #
    #     plt.plot(center[:, 0], center[:, 1], color='grey')
    #     plt.plot(center[:2, 0], center[:2, 1], color='red')
    #
    # plt.xlim(-100,250)
    # plt.ylim(0,350)
    #
    # plt.show()

    lane_graph=build_lane_graph_with_connectors(centerlines)

    edge_graph=build_edge_graph_from_lane_graph_topo(lane_graph,boundary_dict)

    plot_edge_graph(edge_graph, show_nodes=True, show_labels=True)

    #plot_lane_graph(lane_graph)

    # # print(len(polylines))
    #

    centerline_list=[]

    for i,centerline in enumerate(centerlines):
        cur_info = {"id": max_id+1+i}

        cur_info["type"] = 1
        xyz = centerline.centerline

        centerline_list.append(xyz[:,:2])

        cur_polyline = np.concatenate([xyz,np.zeros([len(xyz),1])+cur_info["type"],np.zeros([len(xyz),1])+cur_info["id"]],axis=-1)
        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

        map_infos["lane"].append(cur_info)


    # print(len(line_dict.keys()))

    #plt.show()
    # centerline_list=[]
    # for i,group in enumerate(annotation['lane_line_groups']):
    #
    #     lane1=line_dict[group['lane_line_ids'][0]]
    #     lane2=line_dict[group['lane_line_ids'][1]]
    #
    #     xyz=centerline(lane1, lane2)
    #     cur_info = {"id": max_id+1+i}
    #
    #     cur_info["type"] = 1
    #
    #     centerline_list.append(xyz[:,:2])
    #
    #     cur_polyline = np.concatenate([xyz,np.zeros([len(xyz),1])+cur_info["type"],np.zeros([len(xyz),1])+cur_info["id"]],axis=-1)
    #     cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
    #     polylines.append(cur_polyline)
    #     point_cnt += len(cur_polyline)
    #
    #     map_infos["lane"].append(cur_info)

        # plt.plot(lane1[:,0],lane1[:,1],color='r')
        # plt.plot(lane2[:,0],lane2[:,1],color='b')
        # plt.plot(xyz[:,0],xyz[:,1],color='y')
        # plt.plot(xyz[-2:,0],xyz[-2:,1],color='g')
        # plt.show()
        #
        # print(1)

    map_infos["all_polylines_list"] = polylines
    map_infos["centerline_list"]=centerline_list
    map_infos["lane_graph"]=lane_graph
    map_infos["edge_graph"]=edge_graph.EG
    map_infos["boundary_xyz"]=boundary_dict

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")
    map_infos["all_polylines"] = polylines
    return map_infos