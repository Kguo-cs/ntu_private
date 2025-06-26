from typing import List, Callable, Dict
from collections import deque
from collections import defaultdict
import math
import networkx as nx
from shapely.geometry import LineString, Point, Polygon

from src.light.collision_process import intersecting


class TrafficSystem:
    def __init__(self,light_data):

        light_polyline=light_data["light_polyline"]

        lane_list=[]

        for polyline in light_polyline:
            lane_list.append(LineString(polyline))


        self.group_lights_by_conflicting_lanes(lane_list)


    def group_lights_by_conflicting_lanes(self,light_to_lanes):
        """Group lights if they control conflicting lanes."""
        G = nx.Graph()
        lights = list(light_to_lanes.keys())
        for i in range(len(lights)):
            for j in range(i + 1, len(lights)):
                l1, l2 = lights[i], lights[j]
                if l1.intersecting(l2):  # shared lanes
                    G.add_edge(l1, l2)
        return list(nx.connected_components(G))  # each is a signal group

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
