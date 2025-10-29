import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch

from .desay_lane_graph import build_lane_graph_with_connectors,plot_lane_graph
from .desay_edge_graph import build_edge_graph_from_lane_graph_topo,plot_edge_graph

from .build_centerline import build_centerlines_from_boundaries_xyz


def process(map_infos,map_features,remove_mapid,line_dict,boundary_dict,polylines,point_cnt):
    for mf in map_features:

        id = mf['global_id']

        if id in remove_mapid:
            continue

        feature_data_type = mf['class']
        xyz = np.array(mf['xyz']).T
        cur_info = {"id": id}

        if feature_data_type == "lane_line":
            line_type = mf['attrs']["laneline_type"]
            if line_type == "solid":
                cur_info["type"] = 7
            #  plt.plot(xyz[:, 0], xyz[:, 1], color='r')
            # plt.plot(xyz[:2, 0], xyz[:2, 1], color='b')

            else:  # dot
                cur_info["type"] = 6
                # print(line_type)
                # plt.plot(xyz[:, 0], xyz[:, 1], color='g')
                # plt.plot(xyz[:2, 0], xyz[:2, 1], color='b')
            line_dict[id] = xyz

        elif feature_data_type == "boundary":
            cur_info["type"] = 4
            #
            # if  id in [40,66]:#id ==60 or
            #     # def _arclen2d(xy: np.ndarray) -> np.ndarray:
            #     #     if len(xy) < 2: return np.array([0.0])
            #     #     d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            #     #     return np.concatenate([[0.0], np.cumsum(d)])
            #     #
            #     plt.plot(xyz[:, 0], xyz[:, 1], color='y')
            #     plt.plot(xyz[:2, 0], xyz[:2, 1], color='b')
            #
            boundary_dict[id] = xyz

            # plt.show()

            # print(id,_arclen2d(xyz[:,:2]))

            # print(id)#60,37

        elif feature_data_type == "speed_bump" or feature_data_type == "crosswalk":
            cur_info["type"] = 9
        # elif feature_data_type=="arrow":
        #     continue
        else:
            continue

        cur_polyline = np.concatenate(
            [xyz, np.zeros([len(xyz), 1]) + cur_info["type"], np.zeros([len(xyz), 1]) + cur_info["id"]], axis=-1)

        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
        polylines.append(cur_polyline)
        point_cnt += len(cur_polyline)

        if feature_data_type == "lane_line":
            map_infos["road_line"].append(cur_info)
        elif feature_data_type == "boundary":
            map_infos["road_edge"].append(cur_info)
        elif feature_data_type == "speed_bump" or feature_data_type == "crosswalk":
            map_infos["crosswalk"].append(cur_info)

    return map_infos,line_dict,boundary_dict,polylines,point_cnt


def decode_map_features_from_json(annotation,remove_mapid=[],add_map_object=[]):
    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    line_dict={}
    boundary_dict={}
    polylines = []
    point_cnt = 0

    map_features=annotation['lines']+annotation["traffic_elements"]

    map_infos,line_dict,boundary_dict,polylines,point_cnt=process(map_infos,map_features,remove_mapid,line_dict,boundary_dict,polylines,point_cnt)

    if add_map_object is not None:
        map_infos,line_dict,boundary_dict,polylines,point_cnt=process(map_infos,add_map_object,[],line_dict,boundary_dict,polylines,point_cnt)

    centerlines=build_centerlines_from_boundaries_xyz(boundary_dict)
    ##line_dict,
    # for centerline in centerlines:
    #
    #     center=centerline.centerline
    #
    #     plt.plot(center[:, 0], center[:, 1], color='grey')
    #     plt.plot(center[:2, 0], center[:2, 1], color='red')
    #
    # # plt.xlim(-100,250)
    # #plt.ylim(0,350)
    #
    # plt.xlim(0, 30)
    # plt.ylim(0, 30)
    #
    # plt.show()

    lane_graph=build_lane_graph_with_connectors(centerlines)
    # plot_lane_graph(lane_graph)

    edge_graph=build_edge_graph_from_lane_graph_topo(lane_graph)

    #plot_edge_graph(edge_graph, show_nodes=True, show_labels=True)




    # # print(len(polylines))
    #
    #
    centerline_list=[]
    # #
    # for u, v, data in lane_graph.edges(data=True):
    #     geom = data.get('geom')
    #     if geom is None:
    #         continue
    #     xyz = np.asarray(geom)
    #     kind = data.get('kind', 'lane')
    #     if kind != 'lane':
    #         subtype = data.get('subtype', 'connector')
    #         if subtype in ('longitudinal','turn_left','turn_right') :
    #             cur_info = {"id": 2000 + len(centerline_list)}
    #             cur_info["type"] = 1
    #
    #            # xyz_length=np.linalg.norm(xyz[:,-1]-xyz[:,0],axis=-1)
    #            # if xyz_length>10:
    #             centerline_list.append(xyz[:, :2])
    #             cur_polyline = np.concatenate(
    #                 [xyz, np.zeros([len(xyz), 1]) + cur_info["type"], np.zeros([len(xyz), 1]) + cur_info["id"]],
    #                 axis=-1)
    #             cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
    #             polylines.append(cur_polyline)
    #             point_cnt += len(cur_polyline)
    #
    #             map_infos["lane"].append(cur_info)

    for i,centerline in enumerate(centerlines):
        cur_info = {"id": 1000+i}

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
    map_infos["lane_graph"]=lane_graph
    map_infos["edge_graph"]=edge_graph.edge_graph
    map_infos["boundary_dict"]=boundary_dict
    map_infos["line_dict"]=line_dict

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 8), dtype=np.float32)
        print("Empty polylines.")
    map_infos["all_polylines"] = polylines
    return map_infos