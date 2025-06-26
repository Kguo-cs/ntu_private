
import networkx as nx
from shapely.geometry import LineString, Point, Polygon
import numpy as np
import torch


class TrafficSystem:
    def __init__(self,light_data):

        light_polyline=light_data["light_polyline"]

        lane_list=[]

        for polyline in light_polyline:
            lane_list.append(LineString(polyline.numpy()))


        G=self.group_lights_by_conflicting_lanes(lane_list)

        group_num=len(G)

        conflict_matrix = nx.to_numpy_array(G, nodelist=range(len(light_polyline)), dtype=int)

        #print(nx.connected_components(G))
        # light_idx=light_data["light_idx"]#[:,1]
        # light_num=len(light_idx)
        #
        # conflict_matrix1=np.zeros((light_num,light_num))
        #
        # for i in range(light_num):
        #     for j in range(group_num):
        #         light_i=light_idx[i]
        #         light_j=light_idx[j]
        #         j_1=light_j[light_i==1]
        #         i_1=light_i[light_j==1]
        #
        #         if (len(j_1) and  torch.all(j_1!=1)) or ( len(i_1) and  torch.all(i_1!=1)):
        #             conflict_matrix1[i,j]=1
        #
        # print(1)


    def group_lights_by_conflicting_lanes(self,lights):
        """Group lights if they control conflicting lanes."""
        G = nx.Graph()
        for i in range(len(lights)):
            for j in range(i + 1, len(lights)):
                l1, l2 = lights[i], lights[j]
                if l1.intersects(l2):  # shared lanes
                    G.add_edge(i, j)
        return G

    # def add_light(self, light: TrafficLight):
    #     self.lights[light.name] = light
    #
    # def step(self):
    #     for light in self.lights.values():
    #         light.step()
    #
    # def show_states(self):
    #     for light in self.lights.values():
    #         print(light)
    #
    # def get_history(self):
    #     return {name: list(light.history) for name, light in self.lights.items()}
