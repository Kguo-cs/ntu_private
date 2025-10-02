import os
from math import cos, pi, sin
from multiprocessing import Process

import dearpygui.dearpygui as dpg
from rich import print
from TrafficManager.LimSim.utils.simBase import CoordTF
import cv2
from typing import Dict, List
import matplotlib.pyplot as plt
import torch
import numpy as np
from TrafficManager.LimSim.simModel.DataQueue import (
    CameraImages, ImageQueue, QAQueue, QuestionAndAnswer, RenderQueue,
)
import importlib.util

COLOR_BLACK = (0, 0, 0, 255)
COLOR_WHITE = (255, 255, 255, 255)
COLOR_RED = (255, 0, 0, 255)
COLOR_GREEN = (0, 255, 0, 255)
COLOR_CYAN = (0, 255, 255, 255)
COLOR_MAGENTA = (255, 0, 255, 255)
COLOR_YELLOW = (255, 255, 0, 255)
COLOR_VIOLET = (170, 0, 255, 255)
COLOR_BUTTER = (252, 233, 79, 255)
COLOR_ORANGE = (209, 92, 0, 255)
COLOR_CHOCOLATE = (143, 89, 2, 255)
COLOR_CHAMELEON = (78, 154, 6, 255)
COLOR_SKY_BLUE_0 = (114, 159, 207, 255)
COLOR_SKY_BLUE_1 = (32, 74, 135, 255)
COLOR_PLUM = (92, 53, 102, 255)
COLOR_SCARLET_RED = (164, 0, 0, 255)
COLOR_ALUMINIUM_0 = (238, 238, 236, 255)
COLOR_ALUMINIUM_1 = (211, 215, 207, 255)
COLOR_ALUMINIUM_2 = (66, 62, 64, 255)
COLOR_GREY = (128, 128, 128, 255)

class SequenceError(Exception):
    def __init__(self, errorInfo: str) -> None:
        super().__init__(self)
        self.errorInfo = errorInfo

    def __str__(self) -> str:
        return self.errorInfo


def generateDefaultImage(
        width, height, bgcolor='white',
        text='No Signal', fontcolor='black'
) -> List:
    # 创建一个新的图像
    fig, ax = plt.subplots(figsize=(width / 80, height / 80), dpi=80)

    # 设置背景颜色
    fig.patch.set_facecolor(bgcolor)

    # 移除坐标轴
    ax.axis('off')

    # 在图像中心添加文本
    ax.text(0.5, 0.5, text, fontsize=30, ha='center', va='center', color=fontcolor)

    # 将图像转换为 NumPy 数组
    fig.canvas.draw()
    np_img = np.array(fig.canvas.renderer.buffer_rgba())

    # 关闭图像，释放资源
    plt.close(fig)

    # 归一化并展平图像
    np_img = np_img / 255
    return np_img.flatten().tolist()

# --- Color map ---
color_map = {
    "red": (255, 0, 0, 255),
    "yellow": (255, 255, 0, 255),
    "green": (0, 255, 0, 255),
    "off": (60, 60, 60, 255)
}


shape_symbols = {
    "circle": "O",
    "left_arrow": "<",
    "right_arrow": ">",
    "forward_arrow": "^",
    "uturn": "U"
}

class GUI(Process):
    def __init__(
            self, map_data,data,light,gui_show_static_id
    ) -> None:
        super().__init__()
        self.renderQueue = RenderQueue(1)
        self.imageQueue = ImageQueue(1)

        self.zoom_speed: float = 1.0
        self.is_dragging: bool = False
        self.old_offset = (0, 0)

        # colors
            #[1,4,10]
        # self.lane_style = [
        #     (COLOR_WHITE, 6),  # FREEWAY = 0
        #     (COLOR_ALUMINIUM_2, 6),  # SURFACE_STREET = 1
        #     (COLOR_ORANGE, 6),  # STOP_SIGN = 2
        #     (COLOR_CHOCOLATE, 6),  # BIKE_LANE = 3
        #     (COLOR_SKY_BLUE_1, 4),  # TYPE_ROAD_EDGE_BOUNDARY = 4
        #     (COLOR_PLUM, 4),  # TYPE_ROAD_EDGE_MEDIAN = 5
        #     (COLOR_BUTTER, 2),  # BROKEN = 6
        #     (COLOR_MAGENTA, 2),  # SOLID_SINGLE = 7
        #     (COLOR_SCARLET_RED, 2),  # DOUBLE = 8
        #     (COLOR_CHAMELEON, 4),  # SPEED_BUMP = 9
        #     (COLOR_SKY_BLUE_0, 4),  # CROSSWALK = 10
        # ]
        self.lane_style = [
            (COLOR_WHITE, 6),  # FREEWAY = 0
            (COLOR_ALUMINIUM_2, 2),  # SURFACE_STREET = 1
            (COLOR_ORANGE, 6),  # STOP_SIGN = 2
            (COLOR_CHOCOLATE, 6),  # BIKE_LANE = 3
            (COLOR_RED, 4),  # TYPE_ROAD_EDGE_BOUNDARY = 4
            (COLOR_PLUM, 4),  # TYPE_ROAD_EDGE_MEDIAN = 5
            (COLOR_GREY, 2),  # BROKEN = 6
            (COLOR_WHITE, 2),  # SOLID_SINGLE = 7
            (COLOR_BUTTER, 2),  # DOUBLE = 8
            # (COLOR_CHAMELEON, 4),  # SPEED_BUMP = 9
            (COLOR_GREEN, 4),  # CROSSWALK,SPEED_BUMP = 9
        ]

        self.tl_style = [
            COLOR_RED,  # STOP = 0;
            COLOR_GREEN,  # GO = 1;
            COLOR_YELLOW,  # CAUTION = 2;
            COLOR_GREEN, #COLOR_ALUMINIUM_0,  # NO_LANE_STATE = 3;
            COLOR_GREEN,# COLOR_ALUMINIUM_1,  # LANE_STATE_UNKNOWN = 4;
        ]

        # sdc=0, interest=1, predict=2
        self.agent_role_style = [COLOR_CYAN, COLOR_CHAMELEON, COLOR_MAGENTA]

        #  {0: "vehicle", 1: "pedestrian", 2: "cyclist"}
        self.agent_type_style = [COLOR_ALUMINIUM_0, COLOR_GREEN, COLOR_MAGENTA]

        # make output dir
        # self.save_dir = save_dir
        # self.save_dir.mkdir(exist_ok=True, parents=True)
        #self.mp_xyz=map_infos['all_polylines_list']

        # draw gt
        # self.mp_xyz, self.mp_id, self.mp_type = get_map_features(scenario.map_features)
        #

        self.gui_show_static_id=gui_show_static_id

        self.mp_id = map_data["map_polygon"]["polygon_ids"]

        self.mp_xyz=map_data["map_polygon"]["polygon_xyz"]

        self.mp_type=map_data["map_polygon"]["map_type"]

        position=np.concatenate(self.mp_xyz, axis=0)

        self.netBoundary = ((position[:,0].min()-100, position[:,1].min()-100), (position[:,0].max()+100, position[:,1].max()+100))

        # self.tl_lane_state, self.tl_lane_id = get_traffic_light_features(
        #     scenario.dynamic_map_states
        # )
        # self.tl_lane_id =self.tl_lane_id[step_current]
        self.light=light

        self.ag_size=data["agent"]["shape"]
        ag_role=data["agent"]["role"]
        self.ag_id=data["agent"]["id"].numpy()

        self.routing= {}
        for id, (route,speed) in data["routing"].items():
            self.routing[id]=route.cpu().numpy()

        static_pos, static_yaw, static_size,self.static_type=data["static"]

        self.static = self._get_agent_bbox(np.ones_like(static_yaw[:,0]).astype(np.bool),static_pos, static_yaw, static_size)

        self.static_style = [COLOR_CHOCOLATE, COLOR_CHAMELEON, COLOR_RED,COLOR_BUTTER]

        self.ego_idx=np.where(ag_role[:,0])[0][0]

        self.image_width = 560#767  #
        self.image_height =315 #576#


    def set_ego_pose(self,tokenized_agent,rel_pos,rel_heading):


        initial_pos=tokenized_agent["sampled_pos"][self.ego_idx,1]
        initial_heading=tokenized_agent["sampled_heading"][self.ego_idx,1]

        tokenized_agent["pred_traj_10hz"][self.ego_idx]=initial_pos+rel_pos
        tokenized_agent["pred_head_10hz"][self.ego_idx]=initial_heading+rel_heading



    def setup(self):
        dpg.create_context()
        dpg.create_viewport(
            title="TrafficSimulator",
            width=1800, height=1230)
        dpg.setup_dearpygui()

    def setup_themes(self):
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameRounding, 3,
                    category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameBorderSize, 0.5,
                    category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowBorderSize, 0,
                    category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvNodeCol_NodeBackground, (255, 255, 255)
                )

        dpg.bind_theme(global_theme)

    def create_windows(self):
        with dpg.font_registry():
            font_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), ".", "fonts", "Meslo.ttf"
            )
            default_font = dpg.add_font(font_path, 18)
            self.font2 = dpg.add_font(font_path, 20)

        dpg.bind_font(default_font)

        # Camera Window
        image_width = self.image_width #560
        image_height = self.image_height#315
        ## CAM_FRONT_LEFT
        self.CAM_FRONT_LEFT_TR = dpg.add_texture_registry(show=False)
        dpg.add_dynamic_texture(
            width=self.image_width, height=image_height,
            default_value=generateDefaultImage(image_width, image_height),
            tag='CAM_FRONT_LEFT_TT', parent=self.CAM_FRONT_LEFT_TR
        )
        with dpg.window(tag='CAM_FRONT_LEFT_WINDOW', label='CAM_FRONT_LEFT'):
            dpg.add_image('CAM_FRONT_LEFT_TT')

        ## CAM_FRONT
        self.CAM_FRONT_TR = dpg.add_texture_registry(show=False)
        dpg.add_dynamic_texture(
            width=image_width, height=image_height,
            default_value=generateDefaultImage(image_width, image_height),
            tag='CAM_FRONT_TT', parent=self.CAM_FRONT_TR
        )
        with dpg.window(tag='CAM_FRONT_WINDOW', label='CAM_FRONT'):
            dpg.add_image('CAM_FRONT_TT')

        ## CAM_FRONT_RIGHT
        self.CAM_FRONT_RIGHT_TR = dpg.add_texture_registry(show=False)
        dpg.add_dynamic_texture(
            width=image_width, height=image_height,
            default_value=generateDefaultImage(image_width, image_height),
            tag='CAM_FRONT_RIGHT_TT', parent=self.CAM_FRONT_RIGHT_TR
        )
        with dpg.window(tag='CAM_FRONT_RIGHT_WINDOW', label='CAM_FRONT_RIGHT'):
            dpg.add_image('CAM_FRONT_RIGHT_TT')

        # BEV Window
        dpg.add_window(tag="BEVWindow", label="BEV View")
        dpg.add_draw_node(tag="CanvasBG", parent="BEVWindow")
        dpg.add_draw_node(tag="Canvas", parent="BEVWindow")

        ## Agent Predict BEV Window
        self.PRED_BEV = dpg.add_texture_registry(show=False)
        dpg.add_dynamic_texture(
            width=800, height=800,
            default_value=generateDefaultImage(800, 800),
            tag='PRED_BEV_TT', parent=self.PRED_BEV
        )
        with dpg.window(tag='PredBEVWindow', label='BEV Map'):
            dpg.add_image('PRED_BEV_TT')

    def create_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self.mouse_down)
            dpg.add_mouse_drag_handler(callback=self.mouse_drag)
            dpg.add_mouse_release_handler(callback=self.mouse_release)
            dpg.add_mouse_wheel_handler(callback=self.mouse_wheel)

    def resize_windows(self):
        dpg.set_item_width("CAM_FRONT_LEFT_WINDOW", self.image_width+20)
        dpg.set_item_height("CAM_FRONT_LEFT_WINDOW", self.image_height+45)
        dpg.set_item_pos("CAM_FRONT_LEFT_WINDOW", (10, 10))

        dpg.set_item_width('CAM_FRONT_WINDOW', self.image_width+20)
        dpg.set_item_height('CAM_FRONT_WINDOW', self.image_height+45)
        dpg.set_item_pos('CAM_FRONT_WINDOW', (self.image_width+40, 10))

        dpg.set_item_width('CAM_FRONT_RIGHT_WINDOW', self.image_width+20)
        dpg.set_item_height('CAM_FRONT_RIGHT_WINDOW', self.image_height+45)
        dpg.set_item_pos('CAM_FRONT_RIGHT_WINDOW', (self.image_width*2+70, 10)) #1120

        dpg.set_item_width('BEVWindow', 850)
        dpg.set_item_height('BEVWindow', 850)
        dpg.set_item_pos('BEVWindow', (10, self.image_height+65))

        dpg.set_item_width('PredBEVWindow', 850)
        dpg.set_item_height('PredBEVWindow', 850)
        dpg.set_item_pos('PredBEVWindow', (900, self.image_height+65))


    def drawMainWindowWhiteBG(self):
        pmin, pmax = self.netBoundary
        self.centerx = (pmin[0] + pmax[0]) / 2
        self.centery = (pmin[1] + pmax[1]) / 2
        dpg.draw_rectangle(
            self.ctf.dpgCoord(pmin[0], pmin[1], self.centerx, self.centery),
            self.ctf.dpgCoord(pmax[0], pmax[1], self.centerx, self.centery),
            thickness=0,
            fill=(0, 0, 0),
            parent="CanvasBG"
        )

    def mouse_down(self):
        if not self.is_dragging:
            if dpg.is_item_hovered("BEVWindow"):
                self.is_dragging = True
                self.old_offset = self.ctf.offset

    def mouse_drag(self, sender, app_data):
        if self.is_dragging:
            self.ctf.offset = (
                self.old_offset[0] + app_data[1] / self.ctf.zoomScale,
                self.old_offset[1] + app_data[2] / self.ctf.zoomScale
            )

    def mouse_release(self):
        self.is_dragging = False

    def mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered("BEVWindow"):
            self.zoom_speed = 1 + 0.01 * app_data

    def update_inertial_zoom(self, clip=0.005):
        if self.zoom_speed != 1:
            self.ctf.dpgDrawSize *= self.zoom_speed
            self.zoom_speed = 1 + (self.zoom_speed - 1) / 1.05
        if abs(self.zoom_speed - 1) < clip:
            self.zoom_speed = 1

    @staticmethod
    def _get_agent_bbox(
        agent_valid: np.ndarray,
        agent_pos: np.ndarray,
        agent_yaw: np.ndarray,
        agent_size: np.ndarray,
    ) -> np.ndarray:
        yaw = agent_yaw[agent_valid]  # n, 1
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        v_forward = np.concatenate([cos_yaw, sin_yaw], axis=-1)  # n,2
        v_right = np.concatenate([sin_yaw, -cos_yaw], axis=-1)

        offset_forward = 0.5 * agent_size[agent_valid, 0:1] * v_forward  # [n, 2]
        offset_right = 0.5 * agent_size[agent_valid, 1:2] * v_right  # [n, 2]

        vertex_offset = np.stack(
            [
                -offset_forward + offset_right,
                offset_forward + offset_right,
                offset_forward - offset_right,
                -offset_forward - offset_right,
            ],
            axis=1,
        )  # n,4,2

        agent_pos = agent_pos[agent_valid]
        bbox = agent_pos[:, None, :].repeat(4, 1) + vertex_offset  # n,4,2
        return bbox

    def draw_arrow(self,start, end, color, thickness=1, tip_length=0.2, parent=None):
        # Shaft
        dpg.draw_line(p1=start, p2=end, color=color, thickness=thickness, parent=parent)

        # Arrowhead
        start_np = np.array(start, dtype=np.float32)
        end_np = np.array(end, dtype=np.float32)
        direction = end_np - start_np
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return  # Too short to draw
        unit_dir = direction / length
        ortho = np.array([-unit_dir[1], unit_dir[0]])

        tip_len = tip_length * length
        base = end_np - tip_len * unit_dir
        width = tip_len * 0.5

        left = base + ortho * width
        right = base - ortho * width

        dpg.draw_triangle(
            p1=tuple(left),
            p2=tuple(right),
            p3=tuple(end),
            color=color,
            fill=color,
            parent=parent
        )

    def draw_route(self,node):

        for i,polyline in self.routing.items():
            polyline_tf = self.get_line_tf(polyline[:,:2], self.centerx, self.centery)

            dpg.draw_polyline(
                points=polyline_tf,
                color=(0,0,255),  # RGBA
                thickness=2,
                parent=node
            )

    def draw_static(self,node):
        id=0

        for object,object_type in zip(self.static,self.static_type):

            static_style=self.static_style[object_type]

            bbox_gt1 = self.get_line_tf(object, self.centerx, self.centery)

            dpg.draw_polygon(
                points=bbox_gt1,  # ensure each pt is (x, y)
                color=static_style,  # RGBA tuple, e.g., (255, 0, 0, 255)
                fill=static_style,  # fill with the same color
                thickness=1,  # outline thickness
                parent=node  # your drawing layer
            )
            id = id - 1

            if self.gui_show_static_id:
                center=object.mean(0)

                x=center[0]
                y=center[1]

                dpg.draw_text(
                    self.ctf.dpgCoord(x, y, self.centerx, self.centery),
                    id,
                    color=(255, 255, 255),
                    size=20,
                    parent=node
                )

    def drawVehicles(self, node,_pos,_yaw,ag_type,agent_valid):

        _valid=agent_valid


        _yaw=_yaw[:,None]
        #print(_valid)

        bbox_gt = self._get_agent_bbox(_valid, _pos, _yaw, self.ag_size)
        heading_start = self.get_line_tf(_pos[_valid], self.centerx, self.centery)
        #print(_valid)

        _yaw = _yaw[:, 0][_valid]
        heading_end = self.get_line_tf( _pos[_valid] + 1.5 * np.stack([np.cos(_yaw), np.sin(_yaw)],axis=-1), self.centerx, self.centery)

        _type=ag_type[_valid]

        for i in range(_type.shape[0]):
            id=self.ag_id[i]

            if not _valid[i] or id<0:
               # print(i)
                continue
            if i==self.ego_idx:#[0]
                color=COLOR_CYAN
            else:
                color=self.agent_type_style[_type[i]]
            bbox_gt1=self.get_line_tf( bbox_gt[i], self.centerx, self.centery)

            dpg.draw_polygon(
                points=bbox_gt1,  # ensure each pt is (x, y)
                color=color,  # RGBA tuple, e.g., (255, 0, 0, 255)
                fill=color,  # fill with the same color
                thickness=1,  # outline thickness
                parent=node  # your drawing layer
            )


            # # Draw shaft of the arrow
            if _type[i]==0:
                self.draw_arrow(heading_start[i], heading_end[i], COLOR_BLACK, thickness=1, tip_length=1.0, parent=node)

            center=bbox_gt[i].mean(0)

            x=center[0]
            y=center[1]



            dpg.draw_text(
                self.ctf.dpgCoord(x, y, self.centerx, self.centery),
                id,
                color=(255, 0, 0),
                size=20,
                parent=node
            )

    def get_line_tf(self, line: List[float], ex, ey) -> List[float]:
        return [
            self.ctf.dpgCoord(wp[0], wp[1], ex, ey) for wp in line
        ]

    def drawRoadgraph(self, node):

        for i, _type in enumerate(self.mp_type):
            # if _type==0:
            #     print("freeway")
            #if _type in []:#4,5,6,7,8,9
            color, thickness = self.lane_style[_type]
            polyline = self.mp_xyz[i][:, :2]

            polyline_tf= self.get_line_tf(polyline, self.centerx,self.centery)

            dpg.draw_polyline(
                points=polyline_tf,
                color= color,  # RGBA
                thickness=thickness,
                parent=node
            )

            x = polyline[len(polyline)//2][0]
            y = polyline[len(polyline)//2][1]


            dpg.draw_text(
                self.ctf.dpgCoord(x, y, self.centerx, self.centery),
                self.mp_id[i],
                color=(255, 255, 255),
                size=15,
                parent=node
            )

    # --- Get currently active light(s) ---
    def get_current_active_lights(self,light_group,time):
        if light_group["mode"] == "fixed":
            return [{"color": light_group["fixed_state"]}]

        elif light_group["mode"] == "periodic":
            t =  time % light_group["periodic_schedule"]["cycle_time"]
            total = 0
            for phase in light_group["periodic_schedule"]["phases"]:
                total += phase["duration"]
                if t <= total:
                    return phase["lights_on"]
        elif light_group["mode"] == "custom":
            script_path = light_group["custom_logic"]["script_path"]
            spec = importlib.util.spec_from_file_location("custom_logic", script_path)
            logic = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(logic)
            return logic.get_current_light_state(time)
        return [{"color": "red", "shape": "circle"}]  # fallback

    # --- Draw the traffic light on the canvas ---
    def draw_traffic_light(self,node,time_step):

        if self.light is not None:

            for light_group in self.light:
                orientation = light_group["orientation"]
                lights = light_group["type"]
                active = self.get_current_active_lights(light_group,time_step//10)

                radius = 15
                padding = 0

                position=light_group["position"]

                polyline_tf = self.get_line_tf([position], self.centerx, self.centery)[0]
                start_x=polyline_tf[0]
                start_y=polyline_tf[1]

                for i, light in enumerate(lights):
                    is_active = any(l["color"] == light["color"]  for l in active)
                    color = color_map[light["color"]] if is_active else color_map["off"]

                    if orientation == "vertical":
                        x, y = start_x, start_y+20 + i * (2 * radius + padding)
                    else:
                        x, y = start_x +20+ i * (2 * radius + padding), start_y

                    dpg.draw_circle(center=[x, y], radius=radius, color=(0, 0, 0, 255),
                                    fill=color, thickness=2, parent=node)

                    symbol = shape_symbols.get(light["shape"], "?")
                    dpg.draw_text( pos=(x - 5, y - 5), text=symbol, size=20, color=[255, 255, 255, 255],
                                  parent=node
                                  )


    #     for i_tl, _state in enumerate(light_idx):#self.tl_lane_state[step_t]
    #         _lane_id=self.tl_lane_id[i_tl]#
    #         # _lane_id = self.tl_lane_id[step_t][i_tl]
    #         _lane_idx = np.argwhere(self.mp_id == _lane_id).item()
    #       # print(step_t,_lane_idx)
    #
    #         polyline = self.mp_xyz[_lane_idx][:, :2]
    #
    #         polyline_tf = self.get_line_tf(polyline, self.centerx, self.centery)
    #
    #         # Draw polyline in DPG
    #         # dpg.draw_polyline(
    #         #     points=polyline_tf,
    #         #     color=self.tl_style[_state],  # should be an RGBA tuple (r, g, b, a)
    #         #     thickness=3,
    #         #     parent=node
    #         # )
    #
    #         # If traffic light state indicates active (1 to 3), draw a marker at the end
    #         # if 1 <= _state <= 3:
    #         #     x, y = polyline_tf[-1]
    #         #     offset = 10
    #         #     # Draw tilted cross manually using lines
    #         #     dpg.draw_line((x - offset, y - offset), (x + offset, y + offset), color=self.tl_style[_state],
    #         #                   thickness=6,parent=node)
    #         #     dpg.draw_line((x - offset, y + offset), (x + offset, y - offset), color=self.tl_style[_state],
    #         #                   thickness=6,parent=node )

    def showImage(self, cameraImages: CameraImages):
        front_left_image = cameraImages.CAM_FRONT_LEFT / 255
        dpg.set_value(
            'CAM_FRONT_LEFT_TT', front_left_image.flatten().tolist()
        )
        front_image = cameraImages.CAM_FRONT / 255
        dpg.set_value(
            'CAM_FRONT_TT', front_image.flatten().tolist()
        )
        front_right_image = cameraImages.CAM_FRONT_RIGHT / 255
        dpg.set_value(
            'CAM_FRONT_RIGHT_TT', front_right_image.flatten().tolist()
        )
        if hasattr(cameraImages, 'PRED_BEV') and getattr(cameraImages, 'PRED_BEV') is not None:
            pred_bev_img_array = cameraImages.PRED_BEV / 255
            dpg.set_value(
                'PRED_BEV_TT', pred_bev_img_array.flatten().tolist()
            )


    def render_loop(self):
        self.update_inertial_zoom()
        dpg.delete_item("Canvas", children_only=True)
        canvasNode = dpg.add_draw_node(parent="Canvas")
        #self.vp_x, self.vp_y = dpg.get_viewport_pos()
       # self.vp_w, self.vp_h = dpg.get_viewport_width(), dpg.get_viewport_height()

        #print(self.vp_x, self.vp_y, self.vp_w, self.vp_h)

        try:
            agent_pos,agent_head,agent_type,agent_valid,time_step=self.renderQueue.get()

            ego_position=agent_pos[self.ego_idx]

            self.centerx=ego_position[0]
            self.centery=ego_position[1]

            if time_step is not None:
                self.drawRoadgraph(canvasNode)
                self.draw_route(canvasNode)
                self.draw_traffic_light(canvasNode,time_step)
                self.draw_static(canvasNode)
                self.drawVehicles(canvasNode, agent_pos,agent_head,agent_type,agent_valid)
        except TypeError:
            return


        # Handle camera images
      #  try:
        cameraImagesList = self.imageQueue.get(1)
        if cameraImagesList:
            self.showImage(cameraImagesList[0])
        # except :
        #     pass

        dpg.render_dearpygui_frame()
        # try:
        #     QA = self.QAQ.get()
        #     if QA:
        #         self.showQA(QA)
        # except TypeError:
        #     return

    def run(self):
        self.setup()
        self.create_windows()
        self.create_handlers()
        self.resize_windows()
        self.ctf = CoordTF(120, 'BEVWindow')
        dpg.show_viewport()
        self.drawMainWindowWhiteBG()
        while dpg.is_dearpygui_running():
            self.render_loop()
            dpg.render_dearpygui_frame()

    def draw_input(self,data,current_pos):

        traj_pos = data["map_save"]["traj_pos"].cpu().numpy() # [n_pl, 3, 2]
        #traj_theta = data["map_save"]["traj_theta"] # [n_pl]
        type = data["pt_token"]["type"] .cpu().numpy()  # [n_pl]
        # pl_type = data["pt_token"]["pl_type"]  # [n_pl]
        # light_type= data["pt_token"]["light_type"]   # [n_pl]
        # batch = data["pt_token"]["batch"]

        #mask=np.isin(type,np.array([0,1,2,3,4]))

        #type=type[mask]
        #traj_pos=traj_pos[mask]

        ego_position=current_pos[self.ego_idx][None]

        dist=np.linalg.norm(traj_pos[:,0] - ego_position,axis=-1)

        # Mask for dist < 40
        valid_mask = dist < 40

        # Apply mask
        valid_indices = np.where(valid_mask)[0]
        valid_distances = dist[valid_mask]

        # Get top-20 nearest among the valid ones
        if len(valid_distances) > 0:
            sorted_indices = np.argsort(valid_distances)
            top_k = sorted_indices[:20]  # up to 20 closest

            # Map back to original indices in traj_pos
            topk_indices = valid_indices[top_k]
        else:
            topk_indices = np.array([], dtype=int)

        image_size = (800, 800)
        image = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)  # RGBA

        center = np.array([image_size[0] // 2, image_size[1] // 2])

        for i, _type in enumerate(type):
            if i in topk_indices:
                color, thickness = self.lane_style[_type]  # color: (R,G,B,A)
                polyline = (traj_pos[i][:, :2] - ego_position)*10  # shape [3, 2]

                # Flip y-axis and shift origin to center
                polyline_img = polyline.copy()
                polyline_img[:, 1] *= -1  # Flip Y
                polyline_img = polyline_img + center

                polyline_int = np.round(polyline_img).astype(np.int32).reshape((-1, 1, 2))

                overlay = image.copy()
                cv2.polylines(
                    overlay,
                    [polyline_int],
                    isClosed=False,
                    color=color,  # RGBA
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )

                alpha = color[3] / 255.0
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)

        return image

    def limsim2diffusion(
            self,
            agent_pos,
            agent_heading,
            agent_type,
            data_template,
            agent_command=2,
            last_pose=torch.eye(4),
            drivable_mask=np.ones((200, 200), dtype=np.uint8),
            accel=[0, 0, 9.81],
            rotation_rate=[0, 0, 0],
            vel=[5, 0, 0],
            gen_location="singapore-onenorth",
            gen_prompts="daytime, cloudy, downtown, gray buildings, white cars",
    ):

        ego_pos=agent_pos[self.ego_idx]
        ego_yaw=agent_heading[self.ego_idx]

        ego_x=ego_pos[0]
        ego_y=ego_pos[1]

        bbox_list = []
        label_list = []

        def transform(pos, origin):
            # pos is the coordinate and orientation to be transformed, origin is the coordinate and orientation of the new origin
            # Returns the transformed coordinate and orientation
            x, y, yaw = pos
            x0, y0, yaw0 = origin
            # Calculate the displacement and angle relative to the new origin
            dx = x - x0
            dy = y - y0
            dtheta = yaw - yaw0
            # Calculate the coordinates and orientation in the new coordinate system
            x_new = dx * np.cos(yaw0) + dy * np.sin(yaw0)
            y_new = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
            yaw_new = dtheta
            return x_new, y_new, yaw_new

        for i in range(len(agent_pos)):

            if i != self.ego_idx:
                sur_x = agent_pos[i,0]
                sur_y = agent_pos[i,1]
                sur_yaw = agent_heading[i]

                shape=self.ag_size[i]#length, width, height

                tran_x, tran_y, tran_yaw = transform(
                    (sur_x, sur_y, sur_yaw), (ego_x, ego_y, ego_yaw)
                )
                # tran_x, tran_y, tran_yaw = transform(
                #     (tran_x, tran_y, tran_yaw), (0, 0, -np.pi / 2)
                # )
                # print(sur_veh['id'], tran_x, tran_y, tran_yaw,  tran_yaw+np.pi/2)
                bbox_list.append(
                    [
                        tran_x,
                        tran_y,
                        shape[2]/2,
                        shape[0],
                        shape[1],
                        shape[2],
                        tran_yaw, #-(tran_yaw + np.pi / 2),
                        0,
                        0,
                    ]
                )

                # plot_vehicle((tran_x, tran_y, tran_yaw), color='blue')
                label_list.append(0)  # 0 for vehicle[agent_type[i]]

        send_data = {}
        # ------------ meta ------------ #
        send_data["metas"] = data_template["metas"]
        send_data["metas"]["location"] = gen_location
        send_data["metas"]["description"] = gen_prompts
        # print(
        #     f"location: {send_data['metas']['location']}\ndescription: {send_data['metas']['description']}")
        send_data["metas"]["ego_pos"] = torch.Tensor(
            [
                [np.cos(ego_yaw), -np.sin(ego_yaw), 0, ego_x],
                [np.sin(ego_yaw), np.cos(ego_yaw), 0, ego_y],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        send_data["metas"]["accel"] = accel
        send_data["metas"]["rotation_rate"] = rotation_rate
        send_data["metas"]["vel"] = vel

        # ------------ bboxes ------------ #
        if len(bbox_list) != 0:
            gt_bboxes_3d = torch.tensor(bbox_list)
            send_data["gt_bboxes_3d"] = gt_bboxes_3d
            send_data["gt_labels_3d"] = torch.tensor(label_list)
        else:
            gt_bboxes_3d = torch.empty(0, 9)
            send_data["gt_bboxes_3d"] = gt_bboxes_3d
            send_data["gt_labels_3d"] = torch.empty(0)

        gt_vecs_label=[]
        gt_map_pts=[]
        type_dict={9:0,5:1,6:1,7:1,8:1,4:2}#6 is dash line

        self.patch_size=[100, 100]

        for i, _type in enumerate(self.mp_type):
            if _type in [4,5,6,7,8,9]:#['divider', 'ped_crossing', 'boundary']
                polyline = self.mp_xyz[i][:, :2]

                tran_x, tran_y, tran_yaw = transform(
                    (polyline[:,0], polyline[:,1], 0), (ego_x, ego_y, ego_yaw)
                )
                # tran_x, tran_y, tran_yaw = transform(
                #     (tran_x, tran_y, tran_yaw), (0, 0, -np.pi / 2)
                # )

                x_mask=(tran_x>-self.patch_size[0]//2) & (tran_x<self.patch_size[0]//2)
                y_mask=(tran_x>-self.patch_size[1]//2) & (tran_y<self.patch_size[1]//2)
                mask=x_mask & y_mask

                if mask.any():#any point intersect
                    pts=np.stack([tran_x, tran_y], axis=-1)
                    gt_map_pts.append(pts)
                    gt_vecs_label.append(type_dict[_type])

        send_data["gt_vecs_label"] = gt_vecs_label#type [0,1,2]
        send_data["gt_lines_instance"] = gt_map_pts#list of list 2

        # ---------------ref pose------------------#
        send_data["relative_pose"] = torch.matmul(
            torch.inverse(send_data["metas"]["ego_pos"]), last_pose
        )

        # ---------------drivable mask- -----------------#
        send_data["drivable_mask"] = drivable_mask

        # ---------------Agent command-----------------#
        send_data["agent_command"] = agent_command

        return send_data
