import base64
from datetime import datetime
import io
import json
import os
import sys
import time
import math
from typing import Dict, List, Optional, Tuple
import dearpygui.dearpygui as dpg
from matplotlib import pyplot as plt
import requests
import numpy as np
import torch
import cv2
from PIL import Image
from io import BytesIO
import yaml
# Add LimSim to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "LimSim"))  # noqa
from TrafficManager.utils.sim_utils import limsim2diffusion, normalize_angle, transform_to_ego_frame, interpolate_traj
from TrafficManager.utils.map_utils import VectorizedLocalMap
from LimSim.utils.trajectory import Trajectory, State
from LimSim.trafficManager.traffic_manager import TrafficManager
from LimSim.simModel.MPGUI import GUI
from LimSim.simModel.Model import Model
from LimSim.simModel.DataQueue import CameraImages
from TrafficManager.utils.matplot_render import MatplotlibRenderer
from LimSim.simInfo.CustomExceptions import CollisionChecker, OffRoadChecker
from TrafficManager.utils.scorer import Scorer

from TrafficManager.utils.map_utils import (
    LiDARInstanceLines,
    VectorizedLocalMap,
    project_box_to_image,
    project_map_to_image,
    visualize_bev_hdmap,
    to_tensor
)

class SimulationManager:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_constants()
        self.setup_paths()
        self.model: Optional[Model] = None
        self.planner: Optional[TrafficManager] = None
        self.vectorized_map: Optional[VectorizedLocalMap] = None
        self.gui: Optional[GUI] = None
        self.renderer: Optional[MatplotlibRenderer] = None
        self.checkers: List = []
        self.scorer: Optional[Scorer] = None
        self.timestamp: float = -0.5
        self.data_template: Optional[torch.Tensor] = None
        self.last_pose: torch.Tensor = torch.eye(4)
        self.accel: List[float] = [0, 0, 9.80]
        self.rotation_rate: List[float] = [0, 0, 0]
        self.vel: List[float] = [0, 0, 0]
        self.agent_command: int = 2  # Defined by UniAD  0: Right 1:Left 2:Forward
        self.result_path = f"./results/{datetime.now().strftime('%m-%d-%H%M%S')}/"
        self.img_save_path = f"{self.result_path}imgs/"
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.img_save_path, exist_ok=True)


        self.map_classes= ['divider', 'ped_crossing', 'boundary']
        self.object_classes=['vehicle']
        self.map_bound = {'x':[-50.0, 50.0, 0.5],"y":[-50.0, 50.0, 0.5]}

        xbound = self.map_bound['x']
        ybound = self.map_bound['y']
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)

        self.lidar2img = {
            'CAM_FRONT': torch.tensor([[1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
                                   [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                                   [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
                                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'CAM_FRONT_LEFT': torch.tensor([[6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
                                        [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                        [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'CAM_FRONT_RIGHT': torch.tensor([[1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
                                         [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                                         [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                                         [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'CAM_BACK': torch.tensor([[-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
                                  [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                                  [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
                                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
            'CAM_BACK_LEFT': torch.tensor([[-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
                                       [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                       [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),

            'CAM_BACK_RIGHT': torch.tensor([[3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
                                        [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                                        [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                                        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        }
        self.lidar2cam = {
            'CAM_FRONT': torch.tensor([[1., 0., 0., 0.],
                                   [0., 0., -1., -0.24],
                                   [0., 1., 0., -1.19],
                                   [0., 0., 0., 1.]]),
            'CAM_FRONT_LEFT': torch.tensor([[0.57357644, 0.81915204, 0., -0.22517331],
                                        [0., 0., -1., -0.24],
                                        [-0.81915204, 0.57357644, 0., -0.82909407],
                                        [0., 0., 0., 1.]]),
            'CAM_FRONT_RIGHT': torch.tensor([[0.57357644, -0.81915204, 0., 0.22517331],
                                         [0., 0., -1., -0.24],
                                         [0.81915204, 0.57357644, 0., -0.82909407],
                                         [0., 0., 0., 1.]]),
            'CAM_BACK': torch.tensor([[-1., 0., 0., 0.],
                                  [0., 0., -1., -0.24],
                                  [0., -1., 0., -1.61],
                                  [0., 0., 0., 1.]]),
            'CAM_BACK_LEFT': torch.tensor([[-0.34202014, 0.93969262, 0., -0.25388956],
                                       [0., 0., -1., -0.24],
                                       [-0.93969262, -0.34202014, 0., -0.49288953],
                                       [0., 0., 0., 1.]]),

            'CAM_BACK_RIGHT': torch.tensor([[-0.34202014, -0.93969262, 0., 0.25388956],
                                        [0., 0., -1., -0.24],
                                        [0.93969262, -0.34202014, 0., -0.49288953],
                                        [0., 0., 0., 1.]])
        }
        self.lidar2ego = torch.tensor([[0., 1., 0., -0.39],
                                   [-1., 0., 0., 0.],
                                   [0., 0., 1., 1.84],
                                   [0., 0., 0., 1.]])

        self.camera_intrinsics = {
            cam: self.lidar2img[cam][:3, :3]
            for cam in self.lidar2img
        }
        self.camera2ego = {
            cam: torch.linalg.inv(self.lidar2cam[cam]) @ self.lidar2ego
            for cam in self.lidar2cam
        }

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def setup_constants(self):
        self.DIFFUSION_SERVER = self.config['servers']['diffusion']
        self.DRIVER_SERVER = self.config['servers']['driver']
        self.USE_AGENT_PATH =  self.config['simulation']['use_agent_path']
        self.STEP_LENGTH = self.config['simulation']['step_length']
        self.GUI_DISPLAY = self.config['simulation']['gui_display']
        self.MAX_SIM_TIME = self.config['simulation']['max_sim_time']
        self.EGO_ID = self.config['simulation']['ego_id']
        self.MAP_NAME = self.config['map']['name']
        self.GEN_PROMPT = self.config['map']['gen_description']
        self.IMAGE_SIZE = self.config['image']['size']
        self.TARGET_SIZE = tuple(self.config['image']['target_size'])

    def setup_paths(self):
        data_root = os.path.dirname(os.path.abspath(__file__))
        self.SUMO_CFG_FILE = os.path.join(
            data_root, self.config['map']['sumo_cfg_file'].format(map_name=self.MAP_NAME))
        self.SUMO_NET_FILE = os.path.join(
            data_root, self.config['map']['sumo_net_file'].format(map_name=self.MAP_NAME))
        self.SUMO_ROU_FILE = os.path.join(
            data_root, self.config['map']['sumo_rou_file'].format(map_name=self.MAP_NAME))
        self.DATA_TEMPLATE_PATH = os.path.join(
            data_root, self.config['data']['template_path'])
        self.NU_SCENES_DATA_ROOT = os.path.join(
            data_root, self.config['data']['nu_scenes_root'].format(map_name=self.MAP_NAME))

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def send_request_diffusion(self, diffusion_data: Dict) -> Optional[np.ndarray]:
        serialized_data = {
            k: v.numpy().tolist() if isinstance(v, torch.Tensor) else
            {k2: v2.numpy().tolist() if isinstance(v2, torch.Tensor) else v2 for k2, v2 in v.items()} if isinstance(v, dict) else
            v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in diffusion_data.items()
        }

        try:
            print(f"Sending data to WorldDreamer server...")
            response = requests.post(
                self.DIFFUSION_SERVER + "dreamer-api/", json=serialized_data)
            if response.status_code == 200 and 'image' in response.headers['Content-Type']:
                image = Image.open(BytesIO(response.content))
                images_array = np.array(np.split(np.array(image), 6, axis=0))
                combined_image = np.vstack(
                    (np.hstack(images_array[:3]), np.hstack(images_array[3:])))
                cv2.imwrite(f"{self.img_save_path}diffusion_{str(int(self.timestamp*2)).zfill(3)}.jpg",
                            cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
                return np.array(np.split(np.array(image), 6, axis=0))
        except requests.exceptions.RequestException as e:
            print(f"Warning: Request failed due to {e}")
        return None

    def get_drivable_mask(self, model: Model) -> np.ndarray:
        img = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
        roadgraphRenderData, VRDDict = model.renderQueue.get()
        egoVRD = VRDDict['egoCar'][0]
        ex, ey, ego_yaw = egoVRD.x, egoVRD.y, egoVRD.yaw

        OffRoadChecker().draw_roadgraph(img, roadgraphRenderData, ex, ey, ego_yaw)
        return img.astype(bool)

    def initialize_simulation(self):
        # Initialising models, planners, maps etc
        self.model = Model(
            egoID=self.EGO_ID, netFile=self.SUMO_NET_FILE, rouFile=self.SUMO_ROU_FILE,
            cfgFile=self.SUMO_CFG_FILE, dataBase=self.result_path+"limsim.db", SUMOGUI=False,
            CARLACosim=False,
        )
        self.model.start()
        self.planner = TrafficManager(
            self.model, config_file_path='./LimSim/trafficManager/config.yaml')

        print(f"Testing connection to WorldDreamer & Driver servers...")
        ##requests.get(self.DIFFUSION_SERVER + "dreamer-clean/")
        ##requests.get(self.DRIVER_SERVER + "driver-clean/")

        self.data_template = torch.load(self.DATA_TEMPLATE_PATH)
        self.vectorized_map = VectorizedLocalMap(
            dataroot=self.NU_SCENES_DATA_ROOT, map_name=self.MAP_NAME, patch_size=[100, 100], fixed_ptsnum_per_line=-1)

        self.gui = GUI(self.model)
        if self.GUI_DISPLAY:
            self.gui.start()

        self.renderer = MatplotlibRenderer()
        self.checkers = [OffRoadChecker(), CollisionChecker()]

    def project_bev2img(self,drivable_mask, gt_vecs_pts_loc,gt_vecs_label,gt_bboxes_3d,gt_labels_3d  ):

        layout_canvas = []
        #map_layout_canvas={}
        #box_layout_canvas={}
        for key in ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']:#front_left_image, front_image, front_right_image
            map_canvas = project_map_to_image(gt_vecs_pts_loc, gt_vecs_label, self.camera_intrinsics[key], self.camera2ego[key])
            #gt_bboxes_3d, gt_labels_3d, intrinsic, extrinsic, image = None
            box_canvas = project_box_to_image(gt_bboxes_3d, gt_labels_3d, self.lidar2img[key], object_classes=self.object_classes)

            box_canvas=np.clip(box_canvas,0,1)

            layout_canvas.append(np.concatenate([1-map_canvas, 1-box_canvas], axis=-1))

            #map_layout_canvas[key]=map_canvas
           # box_layout_canvas[key]=box_canvas
            print(map_canvas.max(),box_canvas.max())
        layout_canvas = np.stack(layout_canvas, axis=0)
       # layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))    # 6, C, H, W
        return layout_canvas

    def vectormap_pipeline(self, gt_vecs_label, gt_lines_instance,drivable_mask):
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''

        gt_vecs_label = to_tensor(gt_vecs_label)
        gt_map_pts = []
        for i in range(len(gt_lines_instance)):
            pts = np.array(gt_lines_instance[i])
            gt_map_pts.append(pts)

        gt_vecs_pts_loc=gt_map_pts

        bev_map = visualize_bev_hdmap(gt_vecs_pts_loc, gt_vecs_label, self.canvas_size,
                                      num_classes=3, bound=self.map_bound['x'],
                                      drivable_mask=drivable_mask)

        #bev_map = bev_map.transpose(2, 0, 1)  # C, H, W

        return bev_map,gt_vecs_pts_loc

    def process_frame(self):
        # Single frame processing logic
        if self.scorer is None:
            self.scorer = Scorer(self.model, map_name=self.MAP_NAME,
                                 save_file_path=self.result_path+"drive_arena.pkl")
        try:
            for checker in self.checkers:
                checker.check(self.model)
        except Exception as e:
            print(
                f"WARNING: Checker failed @ timestep {self.model.timeStep}. {e}")
            raise e

        drivable_mask = self.get_drivable_mask(self.model)
        if self.model.timeStep % 5 == 0:
            self.timestamp += 0.5
            if self.timestamp >= self.MAX_SIM_TIME:
                print("Simulation time end.")
                return False

            limsim_trajectories = self.planner.plan(
                self.model.timeStep * 0.1, self.roadgraph, self.vehicles)
            if not limsim_trajectories[self.EGO_ID].states:
                return True

            traj_len = min(
                len(limsim_trajectories[self.EGO_ID].states) - 1, 25)
            local_x, local_y, local_yaw = transform_to_ego_frame(
                limsim_trajectories[self.EGO_ID].states[0], limsim_trajectories[self.EGO_ID].states[traj_len])
            self.agent_command = 2 if local_x <= 5.0 else (
                1 if local_y > 4.0 else 0 if local_y < -4.0 else 2)
            print("Agent command:", self.agent_command)

            diffusion_data = limsim2diffusion(
                self.vehicles, self.data_template, self.vectorized_map, self.MAP_NAME, self.agent_command, self.last_pose, drivable_mask,
                self.accel, self.rotation_rate, self.vel,
                gen_location=self.MAP_NAME,
                gen_prompts=self.GEN_PROMPT,
            )
            #dict_keys(['metas', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_vecs_label', 'gt_lines_instance', 'relative_pose', 'drivable_mask', 'agent_command'])

            gt_vecs_label=diffusion_data["gt_vecs_label"]
            gt_lines_instance=diffusion_data["gt_lines_instance"]

            bev_map,gt_vecs_pts_loc=self.vectormap_pipeline(gt_vecs_label, gt_lines_instance,drivable_mask)

            gt_bboxes_3d=diffusion_data["gt_bboxes_3d"]
            gt_labels_3d=diffusion_data["gt_labels_3d"]


            gen_images=self.project_bev2img(drivable_mask, gt_vecs_pts_loc,gt_vecs_label,gt_bboxes_3d,gt_labels_3d)

            #print(1)
            #self.last_pose = diffusion_data['metas']['ego_pos']
            # gen_images = self.send_request_diffusion(diffusion_data)
            #
            if gen_images is not None:
                front_left_image, front_image, front_right_image = [
                    Image.fromarray(img*255).convert('RGBA') for img in gen_images[:3]]
            else:
                raise ValueError("No images generated!")

            new_width, new_height = self.TARGET_SIZE[0], int(
                (self.TARGET_SIZE[0] / front_image.width) * front_image.height)
            resized_images = [img.resize((new_width, new_height), Image.Resampling.LANCZOS) for img in [
                front_left_image, front_image, front_right_image]]

            ci = CameraImages()
            ci.CAM_FRONT_LEFT, ci.CAM_FRONT, ci.CAM_FRONT_RIGHT = [
                np.array(img) for img in resized_images]
            #print("Current timestamp:", self.timestamp)

            # response = requests.get(self.DRIVER_SERVER + "driver-get/")
            # while response.status_code != 200 or response.text == "false":
            #     # print("The Driver Agent not processing done, try again in 1s")
            #     time.sleep(0.5)
            #     response = requests.get(self.DRIVER_SERVER + "driver-get/")
            #     # print("Driver Agent", response.status_code)
            #
            # driver_output = json.loads(response.text)
            # path_points = driver_output["bbox_results"][0]["planning_traj"][0]
            # print("Driver Agent's Path:", path_points)


            # add driver predict BEV
            # pred_bev_base64 = driver_output["bev_pred_img"]
            # pred_bev_img = base64.b64decode(pred_bev_base64)
            # pred_bev_img = Image.open(io.BytesIO(pred_bev_img))
            # # save image
            # pred_bev_img = pred_bev_img.convert('RGB')
            # pred_bev_img.save(
            #     f"{self.img_save_path}agent_{str(int(self.timestamp*2)).zfill(3)}.jpg")

            pred_bev_img=Image.fromarray(bev_map*255)
            pred_bev_img = pred_bev_img.convert('RGBA')
            pred_bev_img = pred_bev_img.resize(
                (800, 800), Image.Resampling.LANCZOS)
            ci.PRED_BEV = np.array(pred_bev_img, dtype=np.float32)

            self.model.imageQueue.put(ci)
            #
            # path_points.insert(0, [0.0, 0.0])
            # ego_vehicle = self.vehicles['egoCar']
            # ego_traj = interpolate_traj(ego_vehicle, path_points)
            #
            # if len(limsim_trajectories[self.EGO_ID].states) < 10:
            #     yaw_rate = 0
            # else:
            #     yaw_rate = limsim_trajectories[self.EGO_ID].states[9].yaw - \
            #         limsim_trajectories[self.EGO_ID].states[0].yaw
            # vx_1, vx_2 = path_points[2][0] - \
            #     path_points[0][0], path_points[3][0] - path_points[1][0]
            # vy_1, vy_2 = path_points[2][1] - \
            #     path_points[0][1], path_points[3][1] - path_points[1][1]
            # ax, ay = (vx_2 - vx_1) / 0.5, (vy_2 - vy_1) / 0.5
            # self.accel = [ax, ay, 9.80]
            # self.rotation_rate = [0, 0, yaw_rate]
            # self.vel = [limsim_trajectories[self.EGO_ID].states[0].vel, 0, 0]
            # print("Accel:", self.accel, "\nRotation rate:",
            #       self.rotation_rate, "\nVel:", self.vel)

            self.model.putRenderData()
            roadgraphRenderData, VRDDict = self.model.renderQueue.get()
            self.renderer.render(roadgraphRenderData, VRDDict,
                                 f'{self.img_save_path}bev_{str(int(self.timestamp*2)).zfill(3)}.png')

            # self.scorer.record_frame(drivable_mask, is_planning_frame=True,
            #                          planned_traj=ego_traj, ref_traj=limsim_trajectories[self.EGO_ID])

            # limsim_trajectories = {}
            # if self.USE_AGENT_PATH and self.timestamp > 1.0:
            #     ## Because first two frames, drive agents may not ready
            #     print(f"Use agent path to drive.")
            #     limsim_trajectories[self.EGO_ID] = ego_traj
            self.model.setTrajectories(limsim_trajectories)
        else:
            if self.scorer is not None:
                self.scorer.record_frame(
                    drivable_mask, is_planning_frame=False)

        return True

    def run_simulation(self):
        self.initialize_simulation()
        try:
            while not self.model.tpEnd:
                self.model.moveStep()
                self.roadgraph, self.vehicles = self.model.exportSce()
                if self.vehicles and 'egoCar' in self.vehicles:
                    if not self.process_frame():
                        break
                self.model.updateVeh()
        finally:
            self.cleanup()

    def cleanup(self):
        print("Simulation ends")
        if self.scorer:
            self.scorer.save()
        self.model.destroy()
        self.gui.terminate()
        self.gui.join()


def main():
    sim_manager = SimulationManager('config.yaml')
    sim_manager.run_simulation()


if __name__ == '__main__':
    main()
