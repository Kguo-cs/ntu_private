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
from src.smart.model.smart import SMART

# Add LimSim to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "LimSim"))  # noqa
from TrafficManager.utils.map_utils import VectorizedLocalMap
from LimSim.utils.trajectory import Trajectory, State
from LimSim.trafficManager.traffic_manager import TrafficManager
from LimSim.simModel.DataQueue import CameraImages
from torch_geometric.data import HeteroData
from pathlib import Path

from TrafficManager.utils.map_utils import (
    LiDARInstanceLines,
    VectorizedLocalMap,
    project_box_to_image,
    project_map_to_image,
    visualize_bev_hdmap,
    to_tensor
)
import hydra
from waymo_open_dataset.protos import scenario_pb2
import tensorflow as tf
from src.my_data_preprocess import (decode_tracks_from_proto,decode_map_features_from_proto,
                                    decode_dynamic_map_states_from_proto,process_dynamic_map,get_map_features,get_agent_features,process_light,preprocess_map)
from waymo.waymo_render import WaymoRenderer
from waymo.waymo_model import Model
from waymo.waymo_gui import GUI
from time import sleep
import mss
from src.smart.utils import (
    angle_between_2d_vectors,
    transform_to_global,
    weight_init,
    wrap_angle,
)
from waymo.waymo_traffic_light_system import  TrafficSystem

class SimulationManager:
    def __init__(self, cfg,config_path: str) -> None:
        self.config = self.load_config(config_path)
        self.setup_constants()
        self.setup_paths()
        self.setup_planner(cfg)

        #self.result_path = f"./results/{datetime.now().strftime('%m-%d-%H%M%S')}/"
        #self.img_save_path = f"{self.result_path}imgs/"

        #os.makedirs(self.result_path, exist_ok=True)
        #os.makedirs(self.img_save_path, exist_ok=True)

        self.map_classes= [ 'ped_crossing','divider', 'boundary']#green, blue,red
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
            cam:self.lidar2ego @  torch.linalg.inv(self.lidar2cam[cam])
            for cam in self.lidar2cam
        }
#camera2ego = lidar2ego @ lidar2cam[cam]

        # self.camera_intrinsics = self.lidar2img[:, :3]
        # cam2lidar = np.linalg.inv(self.lidar2cam)
        # self.camera2ego = self.lidar2ego @ cam2lidar

        self.initial_step=10

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def initialize_simulation(self,scenario,data):
        # Initialising models, planners, maps etc

        self.gui = GUI(scenario,data,self.initial_step)
        if self.GUI_DISPLAY:
            self.gui.start()

        print(f"Testing connection to WorldDreamer & Driver servers...")
        ##requests.get(self.DIFFUSION_SERVER + "dreamer-clean/")
        ##requests.get(self.DRIVER_SERVER + "driver-clean/")

        self.data_template = torch.load(self.DATA_TEMPLATE_PATH)

        # self.renderer = WaymoRenderer(scenario)
        self.timestamp = self.initial_step
        self.MAX_SIM_TIME = 91

        self.recording = True

        self.record_path = "./results/video/"+str(scenario.scenario_id)+".mp4"
        self.record_fps=10
        self.record_width = 1800  # Must match BEV window texture width
        self.record_height = 1230  # Must match BEV window texture height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' or 'avc1'

        self.video_writer = cv2.VideoWriter(
            self.record_path,
            fourcc,
            self.record_fps,
            (self.record_width, self.record_height)
        )


    def project_bev2img(self,drivable_mask, gt_vecs_pts_loc,gt_vecs_label,gt_bboxes_3d,gt_labels_3d  ):

        layout_canvas = []
        images=[]

        for key in ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT']:#front_left_image, front_image, front_right_image
            image = np.ones((900, 1600, 3), dtype=np.uint8) * 255  # White RGB image

            map_canvas = project_map_to_image(gt_vecs_pts_loc, gt_vecs_label, self.camera_intrinsics[key],self.camera2ego[key],self.lidar2img[key].numpy(),image=image,num_classes=len(self.map_classes), drivable_mask=drivable_mask)
            box_canvas= project_box_to_image(gt_bboxes_3d, gt_labels_3d, self.lidar2img[key], image=image,object_classes=self.object_classes)

            layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))

            images.append(image)

        layout_canvas = np.stack(layout_canvas, axis=0)
        layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))    # 6, C, H, W
        return layout_canvas,images

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
        image = np.ones((self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8) * 255  # White RGB image

        bev_map,bev_image = visualize_bev_hdmap(gt_vecs_pts_loc, gt_vecs_label, self.canvas_size,
                                      num_classes=3, bound=self.map_bound['x'],image=image,
                                      drivable_mask=drivable_mask)

        bev_map = bev_map.transpose(2, 0, 1)  # C, H, W

        return bev_map,gt_vecs_pts_loc,bev_image

    def process_frame(self,map_feature,tokenized_agent):
        if self.timestamp >= self.MAX_SIM_TIME:
            print("Simulation time end.")
            return False
        device=tokenized_agent["type"].device
        agent_type=tokenized_agent["type"].cpu().numpy()

        if self.timestamp % 5 == 0:
            agent_pos=tokenized_agent['sampled_pos'][:,self.timestamp//5-1].cpu().numpy()
            agent_heading=tokenized_agent['sampled_heading'][:,self.timestamp//5-1].cpu().numpy()
            #ci = CameraImages()
            #bev_map=self.gui.draw_input(data,agent_pos)

            #ci.PRED_BEV =bev_map

            diffusion_data = self.gui.limsim2diffusion(
                agent_pos,agent_heading,agent_type,self.data_template
            )

            gt_vecs_label=diffusion_data["gt_vecs_label"]
            gt_lines_instance=diffusion_data["gt_lines_instance"]
            drivable_mask=diffusion_data["drivable_mask"]

            bev_map,gt_vecs_pts_loc,bev_image=self.vectormap_pipeline(gt_vecs_label, gt_lines_instance,drivable_mask)

            gt_bboxes_3d=diffusion_data["gt_bboxes_3d"]
            gt_labels_3d=diffusion_data["gt_labels_3d"]

            send_layouts,gen_images=self.project_bev2img(drivable_mask, gt_vecs_pts_loc,gt_vecs_label,gt_bboxes_3d,gt_labels_3d)

            front_left_image, front_image, front_right_image = [
                Image.fromarray(img).convert('RGBA') for img in gen_images[:3]]

            new_width, new_height = self.TARGET_SIZE[0], int(
                (self.TARGET_SIZE[0] / front_image.width) * front_image.height)
            resized_images = [img.resize((new_width, new_height), Image.Resampling.LANCZOS) for img in [
                front_left_image, front_image, front_right_image]]

            ci = CameraImages()

            ci.CAM_FRONT_LEFT, ci.CAM_FRONT, ci.CAM_FRONT_RIGHT = [
                np.array(img) for img in resized_images]

            bev_image = bev_image.transpose(1, 0, 2)
            pred_bev_img = Image.fromarray(bev_image)#bev_map*255
            pred_bev_img = pred_bev_img.convert('RGBA')
            pred_bev_img = pred_bev_img.resize(
                (800, 800), Image.Resampling.LANCZOS)
            ci.PRED_BEV = np.array(pred_bev_img, dtype=np.float32)

            self.gui.imageQueue.put(ci)
            with torch.no_grad():
                pred_dict = self.planner.encoder.agent_encoder.inference( tokenized_agent, map_feature ,step_current_10hz=self.timestamp,n_step_future_10hz=5 )

            tokenized_agent.update(pred_dict)

            #control ego
            # ego_planned_traj=torch.zeros([5,3],device=device)[None]
            # ego_idx=self.gui.ego_idx
            # token_agent_shape=tokenized_agent["token_agent_shape"][ego_idx][None]
            # token_traj=tokenized_agent["token_traj"][ego_idx][None]
            # sampled_idx=self.planner.token_processor.traj_to_idx(ego_planned_traj[:,-1:],token_agent_shape,token_traj)[0]
            #
            # tokenized_agent["sampled_idx"][ego_idx][-1]=sampled_idx
            #
            # pos_a=tokenized_agent["sampled_pos"][ego_idx][None]
            # head_a=tokenized_agent["sampled_heading"][ego_idx][None]
            #
            # pred_traj, pred_head = transform_to_global(
            #     pos_local=ego_planned_traj[:,:,:2],  # [n_agent, 6*4, 2]
            #     head_local=ego_planned_traj[:,:,2],
            #     pos_now=pos_a[:, -2],  # [n_agent, 2]
            #     head_now=head_a[:, -2],  # [n_agent]
            # )
            #
            # tokenized_agent['pred_traj_10hz'][ego_idx]=pred_traj
            # tokenized_agent['pred_head_10hz'][ego_idx]=pred_head
            # tokenized_agent['sampled_pos'][ego_idx][-1]=pred_traj[:,-1]
            # tokenized_agent['sampled_heading'][ego_idx][-1]=pred_head[:,-1]




        # self.gui.set_ego_pose(tokenized_agent,torch.tensor((0,0)).to(device),torch.tensor(0).to(device)) #set agent_pos,agent_head to (0m,0m) relative to the initial position

        pos = tokenized_agent["pred_traj_10hz"]
        heading = tokenized_agent["pred_head_10hz"]

        light_idx = tokenized_agent["light_idx"][:,(self.timestamp-5)//5].cpu().numpy()
        agent_pos=pos[:,self.timestamp%5].cpu().numpy()
        agent_head=heading[:,self.timestamp%5].cpu().numpy()


        self.gui.renderQueue.put((agent_pos, agent_head, agent_type, light_idx,self.timestamp))

        #rendered_image=self.renderer.render( scenario, tokenized_agent,self.timestamp)

        self.timestamp += 1

        sleep(0.1)
        self.capture_viewport_frame()

        return True

    def run_simulation(self):
        input_dir =self.config["data_path"]
        all_scenarios=os.listdir(input_dir)
        all_scenarios.sort()
        i=0
        for scenario in all_scenarios:
            dataset = tf.data.TFRecordDataset(
                [input_dir+'/'+scenario], compression_type="", #num_parallel_reads=3
            )

            for tf_data in dataset:
                i+=1
                if i!=4:
                    continue
                tf_data = tf_data.numpy()
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(bytes(tf_data))


                track_infos = decode_tracks_from_proto(scenario)
                map_infos = decode_map_features_from_proto(scenario.map_features)
                dynamic_map_infos = decode_dynamic_map_states_from_proto(
                    scenario.dynamic_map_states
                )
                current_time_index = scenario.current_time_index

                tf_lights = process_dynamic_map(dynamic_map_infos)
                tf_current_light = tf_lights.loc[tf_lights["time_step"] == current_time_index]
                map_data = get_map_features(map_infos, tf_current_light)

                data = preprocess_map(map_data)

                #add agent
                # track_infos["object_id"]=np.concatenate([track_infos["object_id"],np.zeros([1])])
                # track_infos["valid"]=np.concatenate([track_infos["valid"],np.ones([1,91]).astype(bool)])
                # track_infos["role"]=np.concatenate([track_infos["role"],np.zeros([1,3]).astype(bool)])

                #add static object
                # track_infos["object_type"]=np.concatenate([track_infos["object_type"],np.zeros([1]).astype(int)])
                # state=np.zeros([1,91,9])# x, y, z, length, width, height,heading,vx,vy
                # state[0,:,0]=362
                # state[0,:,1]=6300
                # state[0,:,3]=10
                # state[0,:,4]=10
                # state[0,:,5]=10
                # track_infos["states"]=np.concatenate([track_infos["states"],state])

                #add dynamic ped
                # track_infos["object_type"]=np.concatenate([track_infos["object_type"],np.ones([1]).astype(int)])
                # state=np.zeros([1,91,9])# x, y, z, length, width, height,heading,vx,vy
                #
                # state[0,0,0]=362
                # state[0,5,0]=361
                # state[0,10,0]=360
                # state[0,:,1]=6270
                # state[0,:,3]=1
                # state[0,:,4]=1
                # state[0,:,5]=1
                # state[0,:,6]=-np.pi
                # track_infos["states"]=np.concatenate([track_infos["states"],state])



                #delete agent
                # id=997
                # mask=np.where(track_infos["object_id"]!=id)
                # track_infos["object_id"]=track_infos["object_id"][mask]
                # track_infos["object_type"]=track_infos["object_type"][mask]
                # track_infos["states"]=track_infos["states"][mask]
                # track_infos["valid"]=track_infos["valid"][mask]
                # track_infos["role"]=track_infos["role"][mask]



                data["agent"] = get_agent_features(
                    track_infos,
                    split="validation",
                    num_historical_steps=current_time_index + 1,
                    num_steps=91,
                )
                data["agent"]["batch"]=torch.zeros(data["agent"]["num_nodes"]).long()
                data["pt_token"]["batch"]=torch.zeros(data["pt_token"]["num_nodes"]).long()

                data["light"] = process_light(map_infos, tf_lights, tf_current_light)
                data["light"]["batch"]=torch.zeros(data["light"]["num_nodes"]).long()

                self.initialize_simulation(scenario,data)
                #custom_traffic_Light
               # trafficlight_system=TrafficSystem(data["light"])

                batch_data = HeteroData(data).cuda()
                batch_data.num_graphs=1

                tokenized_map, tokenized_agent = self.planner.token_processor(batch_data)

                map_feature = self.planner.encoder.map_encoder(tokenized_map)

                agent_num=len(tokenized_agent["batch"])

                # #add traffic light
                # light_num=len(tokenized_agent["light_idx"])
                # tokenized_agent["light_idx"]=tokenized_agent["light_idx"][:light_num]#torch.ones_like(tokenized_agent["light_idx"][:light_num]).long()
                # tokenized_agent["pos_lg"]=tokenized_agent["pos_lg"][:light_num]
                # tokenized_agent["orient_lg"]=tokenized_agent["orient_lg"][:light_num]
                # tokenized_agent["batch_lg"]=tokenized_agent["batch_lg"][:light_num]
                # tokenized_agent["valid_mask"]=tokenized_agent["valid_mask"][:light_num+agent_num]


                while True:
                    if not self.process_frame(map_feature, tokenized_agent):
                        break

                self.cleanup()

               # return

    def capture_viewport_frame(self):
        if not self.recording or self.video_writer is None:
            return

        with mss.mss() as sct:
            # Get viewport position & size
            vp_x, vp_y = 140,128
            vp_w, vp_h =1800, 1230

            monitor = {
                "top": int(vp_y),
                "left": int(vp_x),
                "width": int(vp_w),
                "height": int(vp_h),
            }

            # Capture screen region
            sct_img = sct.grab(monitor)
            frame = np.array(sct_img)

            # Convert BGRA to BGR
            frame = frame[:, :, :3]

            # # Resize to match video size
            # frame = cv2.resize(frame, (self.record_width, self.record_height))
            # # Save as image
            # img_path = f"frame_{int(time.time() * 1000)}.png"
            # cv2.imwrite(img_path, frame)
            # print(f"[INFO] Frame saved to {img_path}")
            # print("Frame shape:", frame.shape)

            self.video_writer.write(frame)


    def cleanup(self):
        print("Simulation ends")

        # if self.scorer:
        #     self.scorer.save()
        # if self.gui.video_writer is not None:
        self.gui.terminate()
        self.gui.join()
        self.video_writer.release()

    def setup_constants(self):
        self.DIFFUSION_SERVER = self.config["servers"]["diffusion"]
        self.DRIVER_SERVER = self.config["servers"]["driver"]
        self.STEP_LENGTH = self.config["simulation"]["step_length"]
        self.GUI_DISPLAY = self.config["simulation"]["gui_display"]
        self.MAX_SIM_TIME = self.config["simulation"]["max_sim_time"]
        self.EGO_ID = self.config["simulation"]["ego_id"]
        self.MAP_NAME = self.config["map"]["name"]
        self.IMAGE_SIZE = self.config["image"]["size"]
        self.TARGET_SIZE = tuple(self.config["image"]["target_size"])

    def setup_paths(self):
        data_root = os.path.dirname(os.path.abspath(__file__))
        self.SUMO_CFG_FILE = os.path.join(
            data_root,
            self.config["map"]["sumo_cfg_file"].format(map_name=self.MAP_NAME),
        )
        self.SUMO_NET_FILE = os.path.join(
            data_root,
            self.config["map"]["sumo_net_file"].format(map_name=self.MAP_NAME),
        )
        self.SUMO_ROU_FILE = os.path.join(
            data_root,
            self.config["map"]["sumo_rou_file"].format(map_name=self.MAP_NAME),
        )
        self.DATA_TEMPLATE_PATH = os.path.join(
            data_root, self.config["data"]["template_path"]
        )
        self.NU_SCENES_DATA_ROOT = os.path.join(
            data_root,
            self.config["data"]["nu_scenes_root"].format(map_name=self.MAP_NAME),
        )

    def setup_planner(self,cfg):
        self.planner = SMART(cfg.model.model_config)

        if torch.cuda.is_available():
            state_dict = torch.load(self.config["planner_path"])["state_dict"]
        else:
            state_dict = torch.load(self.config["planner_path"], map_location=torch.device("cpu"))["state_dict"]


        # self.planner.load_state_dict(state_dict)
        self.planner.cuda()
        self.planner.eval()


@hydra.main(config_path="../configs/", config_name="run.yaml", version_base=None)
def main(cfg):
    sim_manager = SimulationManager(cfg, 'waymo/waymo_config.yaml')
    sim_manager.run_simulation()


if __name__ == '__main__':
    main()
