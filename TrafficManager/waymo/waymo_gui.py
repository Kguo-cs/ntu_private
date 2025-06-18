import os
from math import cos, pi, sin
from multiprocessing import Process
from typing import Dict, List, Tuple

import dearpygui.dearpygui as dpg
import numpy as np
from matplotlib import pyplot as plt
from rich import print

from TrafficManager.LimSim.simModel.DataQueue import (
    ERD, JLRD, LRD, RGRD, VRD, CameraImages, QuestionAndAnswer,
)
from TrafficManager.LimSim.simModel.Model import Model
from TrafficManager.LimSim.utils.simBase import CoordTF
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2
import torch

from src.utils.video_recorder import ImageEncoder
import matplotlib.pyplot as plt
from typing import Dict, List
from math import cos, pi, sin
import matplotlib.pyplot as plt
import torch
import numpy as np
from copy import deepcopy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io
from src.utils.vis_waymo import get_map_features,get_traffic_light_features,get_agent_features

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


class GUI(Process):
    def __init__(
            self, model: Model,scenario
    ) -> None:
        super().__init__()
        self.renderQueue = model.renderQueue
        self.imageQueue = model.imageQueue
        self.QAQ = model.QAQ
        if model.netBoundary:
            self.netBoundary = model.netBoundary
        else:
            raise SequenceError(
                'Class `GUI` must be initialized after `model.start()`.'
            )

        self.zoom_speed: float = 1.0
        self.is_dragging: bool = False
        self.old_offset = (0, 0)

        # colors

        self.lane_style = [
            (COLOR_WHITE, 6),  # FREEWAY = 0
            (COLOR_ALUMINIUM_2, 6),  # SURFACE_STREET = 1
            (COLOR_ORANGE, 6),  # STOP_SIGN = 2
            (COLOR_CHOCOLATE, 6),  # BIKE_LANE = 3
            (COLOR_SKY_BLUE_1, 4),  # TYPE_ROAD_EDGE_BOUNDARY = 4
            (COLOR_PLUM, 4),  # TYPE_ROAD_EDGE_MEDIAN = 5
            (COLOR_BUTTER, 2),  # BROKEN = 6
            (COLOR_MAGENTA, 2),  # SOLID_SINGLE = 7
            (COLOR_SCARLET_RED, 2),  # DOUBLE = 8
            (COLOR_CHAMELEON, 4),  # SPEED_BUMP = 9
            (COLOR_SKY_BLUE_0, 4),  # CROSSWALK = 10
        ]

        self.tl_style = [
            COLOR_ALUMINIUM_1,  # STATE_UNKNOWN = 0;
            COLOR_RED,  # STOP = 1;
            COLOR_YELLOW,  # CAUTION = 2;
            COLOR_GREEN,  # GO = 3;
            COLOR_VIOLET,  # FLASHING = 4;
        ]
        # sdc=0, interest=1, predict=2
        self.agent_role_style = [COLOR_CYAN, COLOR_CHAMELEON, COLOR_MAGENTA]

        #  {0: "vehicle", 1: "pedestrian", 2: "cyclist"}
        self.agent_type_style = [COLOR_ALUMINIUM_0, COLOR_CHAMELEON, COLOR_MAGENTA]

        # make output dir
        # self.save_dir = save_dir
        # self.save_dir.mkdir(exist_ok=True, parents=True)

        # draw gt
        self.mp_xyz, self.mp_id, self.mp_type = get_map_features(scenario.map_features)

        self.tl_lane_state, self.tl_lane_id = get_traffic_light_features(
            scenario.dynamic_map_states
        )

        ag_valid, ag_xy, ag_yaw, self.ag_size, self.ag_role, ag_id = get_agent_features(
            scenario, step_current=10
        )
        # self.ag_id2size = dict(zip(ag_id, ag_size))
        # self.ag_id2role = dict(zip(ag_id, ag_role))
        self.ego_idx=np.where(self.ag_role==0)[0][0]


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
        image_width = 560
        image_height = 315
        ## CAM_FRONT_LEFT
        self.CAM_FRONT_LEFT_TR = dpg.add_texture_registry(show=False)
        dpg.add_dynamic_texture(
            width=image_width, height=image_height,
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

        # # Prompts windows
        # dpg.add_window(tag="PromptsWindow", label='Prompts')

        # # Response Window
        # dpg.add_window(tag="ResponseWindow", label='Reasoning and decision')

    def create_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self.mouse_down)
            dpg.add_mouse_drag_handler(callback=self.mouse_drag)
            dpg.add_mouse_release_handler(callback=self.mouse_release)
            dpg.add_mouse_wheel_handler(callback=self.mouse_wheel)

    def resize_windows(self):
        dpg.set_item_width("CAM_FRONT_LEFT_WINDOW", 580)
        dpg.set_item_height("CAM_FRONT_LEFT_WINDOW", 360)
        dpg.set_item_pos("CAM_FRONT_LEFT_WINDOW", (10, 10))

        dpg.set_item_width('CAM_FRONT_WINDOW', 580)
        dpg.set_item_height('CAM_FRONT_WINDOW', 360)
        dpg.set_item_pos('CAM_FRONT_WINDOW', (600, 10))

        dpg.set_item_width('CAM_FRONT_RIGHT_WINDOW', 580)
        dpg.set_item_height('CAM_FRONT_RIGHT_WINDOW', 360)
        dpg.set_item_pos('CAM_FRONT_RIGHT_WINDOW', (1190, 10))

        dpg.set_item_width('BEVWindow', 850)
        dpg.set_item_height('BEVWindow', 850)
        dpg.set_item_pos('BEVWindow', (10, 380))

        dpg.set_item_width('PredBEVWindow', 850)
        dpg.set_item_height('PredBEVWindow', 850)
        dpg.set_item_pos('PredBEVWindow', (900, 380))

        # dpg.set_item_width('PromptsWindow', 470)
        # dpg.set_item_height('PromptsWindow', 800)
        # dpg.set_item_pos('PromptsWindow', (10, 380))

        # dpg.set_item_width('ResponseWindow', 470)
        # dpg.set_item_height('ResponseWindow', 800)
        # dpg.set_item_pos('ResponseWindow', (1300, 380))

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

    def drawVehicles(self, node,_pos,_yaw,ag_type):

        _valid=np.ones_like(_yaw).astype(bool)

        _yaw=_yaw[:,None]

        bbox_gt = self._get_agent_bbox(_valid, _pos, _yaw, self.ag_size)
        heading_start = self.get_line_tf(_pos[_valid], self.centerx, self.centery)

        _yaw = _yaw[:, 0][_valid]
        heading_end = self.get_line_tf( _pos[_valid] + 1.5 * np.stack([np.cos(_yaw), np.sin(_yaw)],axis=-1), self.centerx, self.centery)


        _role = self.ag_role[_valid]
        _type=ag_type[_valid]
        for i in range(_role.shape[0]):
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
            self.draw_arrow(heading_start[i], heading_end[i], COLOR_BLACK, thickness=1, tip_length=1.0, parent=node)

    def get_line_tf(self, line: List[float], ex, ey) -> List[float]:
        return [
            self.ctf.dpgCoord(wp[0], wp[1], ex, ey) for wp in line
        ]

    def drawRoadgraph(self, node):

        for i, _type in enumerate(self.mp_type):
            # if _type==0:
            #     print("freeway")
            if _type in [0,1,2,3,4,10]:
                color, thickness = self.lane_style[_type]
                polyline = self.mp_xyz[i][:, :2]

                polyline_tf= self.get_line_tf(polyline, self.centerx,self.centery)

                dpg.draw_polyline(
                    points=polyline_tf,
                    color= color,  # RGBA
                    thickness=thickness,
                    parent=node
                )

    def draw_traffic_light(self,node,step_t):

        for i_tl, _state in enumerate(self.tl_lane_state[step_t]):
            _lane_id = self.tl_lane_id[step_t][i_tl]
            _lane_idx = np.argwhere(self.mp_id == _lane_id).item()
          # print(step_t,_lane_idx)

            polyline = self.mp_xyz[_lane_idx][:, :2]

            polyline_tf = self.get_line_tf(polyline, self.centerx, self.centery)

            # Draw polyline in DPG
            dpg.draw_polyline(
                points=polyline_tf,
                color=self.tl_style[_state],  # should be an RGBA tuple (r, g, b, a)
                thickness=3,
                parent=node
            )

            # If traffic light state indicates active (1 to 3), draw a marker at the end
            # if 1 <= _state <= 3:
            #     x, y = polyline_tf[-1]
            #     offset = 10
            #     # Draw tilted cross manually using lines
            #     dpg.draw_line((x - offset, y - offset), (x + offset, y + offset), color=self.tl_style[_state],
            #                   thickness=6,parent=node)
            #     dpg.draw_line((x - offset, y + offset), (x + offset, y - offset), color=self.tl_style[_state],
            #                   thickness=6,parent=node )

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
        try:
            agent_pos,agent_head,agent_type,time_step=self.renderQueue.get()

            ego_position=agent_pos[self.ego_idx]

            self.centerx=ego_position[0]
            self.centery=ego_position[1]

            if time_step is not None:
                self.drawRoadgraph(canvasNode)
                self.draw_traffic_light(canvasNode,time_step)
                self.drawVehicles(canvasNode, agent_pos,agent_head,agent_type)
        except TypeError:
            return

        # Handle camera images
        try:
            cameraImagesList = self.imageQueue.get()
            if cameraImagesList:
                self.showImage(cameraImagesList[0])
        except :
            pass

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
        type = data["pt_token"]["type"]  # [n_pl]
        # pl_type = data["pt_token"]["pl_type"]  # [n_pl]
        # light_type= data["pt_token"]["light_type"]   # [n_pl]
        # batch = data["pt_token"]["batch"]

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

        # map_token=token_traj_src[map_token_idx]
        #
        # traj_pos_local, _ = transform_to_local(
        #     pos_global=traj_pos,  # [n_pl, 3, 2]
        #     head_global=None,  # [n_pl, 1]
        #     pos_now=traj_pos[:, 0],  # [n_pl, 2]
        #     head_now=traj_theta,  # [n_pl]
        # )

        return 1