import os
import sys
from typing import Dict
import numpy as np
import torch
import cv2
from PIL import Image
import yaml
torch.set_float32_matmul_precision("highest")#  #“highest” (default),

import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Enforce deterministic ops where PyTorch supports them
torch.use_deterministic_algorithms(True, warn_only=True)

dir_name=os.path.dirname(os.path.abspath(__file__))

# Add LimSim to sys.path
sys.path.append(dir_name)  # noqa

sys.path.append(os.path.dirname(dir_name))  # noqa

sys.path.append(os.path.join(dir_name, "LimSim"))  # noqa
from LimSim.simModel.DataQueue import CameraImages
from torch_geometric.data import HeteroData

from TrafficManager.utils.map_utils import (
    project_box_to_image,
    project_map_to_image,
    visualize_bev_hdmap,
    to_tensor
)
import hydra
from waymo_open_dataset.protos import scenario_pb2
import tensorflow as tf

from src.smart.model.smart_gail1 import SMART_IQ

from src.my_data_preprocess import (decode_tracks_from_proto, decode_map_features_from_proto,
                                    get_map_features, get_agent_features, preprocess_map)
from waymo.waymo_gui import GUI
from time import sleep
from src.smart.utils import (
    wrap_angle, transform_to_local,
)
import json
from omegaconf import OmegaConf
import time
import psutil
# from pynvml import *

from collections import defaultdict
from desay_utils.check_visible import check_occlusion_multi_cam
from desay_utils.desay_data_process import decode_map_features_from_json
from desay_utils.idm_policy import idm_planner
from desay_utils.scene_generator import TrafficGenerator,make_ego_agent
from desay_utils.plot_route import plot_agents_on_map
from desay_utils.static_object_generator import generate_static_elements_from_raw,StaticSpec,plot_static_on_map
from desay_utils.route_utils import compute_yaw_from_traj,nearest_edges_biside,append_segment_with_step

def print_cpu_usage(interval=1.0):
    pid = os.getpid()
    process = psutil.Process(pid)

    # 初次调用时统计间隔 CPU 时间，第二次才有结果
    process.cpu_percent(interval=None)  # 清除旧状态
    time.sleep(interval)
    cpu_usage = process.cpu_percent(interval=None)

    print(f"🧠 当前进程 CPU 占用率：{cpu_usage:.2f}%")

# def get_self_gpu_usage():
#     nvmlInit()
#     pid = os.getpid()
#     device_count = nvmlDeviceGetCount()
#
#     for i in range(device_count):
#         handle = nvmlDeviceGetHandleByIndex(i)
#         procs = nvmlDeviceGetComputeRunningProcesses(handle)
#
#         for p in procs:
#             if p.pid == pid:
#                 meminfo = nvmlDeviceGetMemoryInfo(handle)
#                 util = nvmlDeviceGetUtilizationRates(handle)
#                 print(f"[GPU:{i}] PID={pid} 使用显存: {p.usedGpuMemory / 1024**2:.1f} MiB / {meminfo.total / 1024**2:.1f} MiB")
#                 print(f"[GPU:{i}] GPU 核心利用率: {util.gpu}%")
#                 break
#
#     nvmlShutdown()


def get_process_memory():
    """返回当前进程的 RSS（真实占用内存，单位 MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

class SimulationManager:
    def __init__(self, model_cfg,config: str) -> None:
        self.config = config
        self.setup_planner(model_cfg)
        self.GUI_DISPLAY =self.config["gui_display"]

        self.TARGET_SIZE = tuple(self.config["image"]["target_size"])

        self.map_classes= [ 'ped_crossing','divider', 'boundary']#green, blue,red
        self.object_classes=['vehicle']
        self.map_bound = {'x':[-50.0, 50.0, 0.5],
                          "y":[-50.0, 50.0, 0.5]}

        xbound = self.map_bound['x']
        ybound = self.map_bound['y']
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)

        camera_parameter_path=self.config["camera_parameter_path"]

        with open(camera_parameter_path, 'r') as f:
            data = json.load(f)

        self.lidar2img={}
        self.image_size={}

        for key,item in data.items():
            # Step 1: Load matrices # using the 3x3 'intrinsic' matrix
            K = np.array(item["intrinsic"])
            R = np.array(item["rot"])
            T = np.array(item["tran"])

            R_lidar2cam = R.T
            T_lidar2cam = -R_lidar2cam @ T

            RT = np.concatenate([R_lidar2cam, T_lidar2cam[:,None]], axis=1)
            P= K @ RT
            self.lidar2img[key] = np.vstack([P, np.array([[0, 0, 0, 1]])])  # shape: (4, 4)

            self.image_size[key]=(int(K[1,2]*2),int(K[0,2]*2))

        self.initial_step=10
        self.output_json_path=self.config["output_json_path"]
        self.input_json_path=self.config["input_json_path"]
        self.lidar_height=self.config["lidar_height"]

        self.output_ego_json_path=self.config["output_ego_json_path"]

        self.camera_rendering_time=[]
        self.traffic_model_time=[]
        self.output_time=[]

        self.random_seed=int(self.config["random_seed"])
        self.light = self.config["traffic_lights"]
        self.traffic_signs = self.config["traffic_signs"]

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def initialize_simulation(self,map_data,data):
        # Initialising models, planners, maps etc

        self.gui = GUI(map_data,data,self.light,self.traffic_signs,self.config["gui_show_static_id"])
        if self.GUI_DISPLAY:
            self.gui.start()

        self.timestamp = self.initial_step
        self.MAX_SIM_TIME = self.config["max_sim_time"]+self.initial_step

        self.recording = False

        if self.recording:
            self.record_path = "./results/video/record.mp4"
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
            # image = np.ones((1080,1920, 3), dtype=np.uint8) * 255  # White RGB image
            #image = np.ones((900, 1600, 3), dtype=np.uint8) * 255  # White RGB image
            image = np.ones((self.image_size[key][0],self.image_size[key][1], 3), dtype=np.uint8) * 255  # White RGB image

            map_canvas = project_map_to_image(gt_vecs_pts_loc, gt_vecs_label, self.lidar2img[key],self.lidar_height,image=image,num_classes=len(self.map_classes), drivable_mask=None)
            box_canvas= project_box_to_image(gt_bboxes_3d, gt_labels_3d, self.lidar2img[key],self.lidar_height, image=image,object_classes=self.object_classes)
        #     layout_canvas.append(np.concatenate([map_canvas, box_canvas], axis=-1))
            images.append(image)
        #
        # layout_canvas = np.stack(layout_canvas, axis=0)
        # layout_canvas = np.transpose(layout_canvas, (0, 3, 1, 2))    # 6, C, H, W
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

    def process_frame(self,data,map_feature,tokenized_agent):
        if self.timestamp >= self.MAX_SIM_TIME:
            print("Simulation time end.")
            return False
        agent_type=tokenized_agent["type"].cpu().numpy()
        control_mask = torch.zeros_like(self.agent_mask)

        if self.timestamp % 5 == 0:
            if self.GUI_DISPLAY:
                camera_rendering_start = time.time()
                #rss_before = get_process_memory()

                agent_pos = tokenized_agent['sampled_pos'][:,self.timestamp//5-1].cpu().numpy()
                agent_heading=tokenized_agent['sampled_heading'][:,self.timestamp//5-1].cpu().numpy()

                diffusion_data = self.gui.limsim2diffusion(
                    agent_pos,agent_heading,agent_type
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

                new_width, new_height = self.TARGET_SIZE[0], self.TARGET_SIZE[1]#[560, 315]
                resized_images = [img.resize((new_width, new_height), Image.Resampling.NEAREST) for img in [
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
                # bev_map=self.gui.draw_input(data,agent_pos)
                # ci.PRED_BEV =bev_map

                self.gui.imageQueue.put(ci)

                self.camera_rendering_time.append(time.time()-camera_rendering_start)
                #print(get_process_memory()-rss_before)
               # print(get_self_gpu_usage())
                #print(print_cpu_usage())

            traffic_model_start=time.time()
           # rss_before = get_process_memory()


            with torch.no_grad():
                pred_dict = self.planner.encoder.agent_encoder.inference( tokenized_agent, map_feature ,step_current_10hz=self.timestamp,n_step_future_10hz=5 )

            for key in ["sampled_idx","sampled_pos","sampled_heading","valid_mask","token_mask"]:
                pred_value=pred_dict[key]
                tokenized_agent[key][self.agent_mask,:pred_value.shape[1]] = pred_value[self.agent_mask]

            for key in ["pred_traj_10hz","pred_head_10hz"]:
                tokenized_agent[key][self.agent_mask,self.timestamp+1:self.timestamp+6] = pred_dict[key][self.agent_mask]

            tokenized_agent["all_valid"][self.agent_mask, self.timestamp + 1:self.timestamp + 6] = True

            self.traffic_model_time.append(time.time()-traffic_model_start)
           # print(get_process_memory() - rss_before)
            #print(get_self_gpu_usage())
            #print(print_cpu_usage())


            # for id, (route,speed) in self.route.items():
            #     idx = torch.where(tokenized_agent["id"] == id)[0]
            #
            #     control_mask[idx]=True
            #
            #     all_pos = tokenized_agent["pred_traj_10hz"][:, self.timestamp]
            #
            #     all_heading = tokenized_agent["pred_head_10hz"][:, self.timestamp]
            #
            #     all_shape = tokenized_agent["shape"][:, :2]  # length, width
            #
            #     # route [n,2]
            #     prev_pos = tokenized_agent["pred_traj_10hz"][:, self.timestamp - 1]
            #
            #     all_velocity = (all_pos - prev_pos) / 0.1
            #
            #     new_pos, new_heading = idm_planner(route,  idx, all_pos, all_heading, all_velocity, all_shape,
            #                                        desired_speed=speed)  # plan 0.5 second
            #
            #     tokenized_agent["pred_traj_10hz"][idx, self.timestamp + 1:self.timestamp + 6]=new_pos
            #     tokenized_agent["pred_head_10hz"][idx, self.timestamp + 1:self.timestamp + 6]=new_heading

            #control ego
            if self.input_json_path is not None:
                with open(self.input_json_path, "r") as f:
                    data = json.load(f)

                if len(data)>0:
                    object_id = tokenized_agent["id"]

                    for t in data.keys():
                        ids=data[t]["tracking_id"]
                        bboxes=torch.tensor(data[t]["bboxes"]).cuda()
                        for id,box in zip(ids,bboxes):
                            obj_mask=id==object_id
                            control_mask[obj_mask] = True

                            tokenized_agent['pred_traj_10hz'][obj_mask,int(t)]=box[:2]
                            tokenized_agent['pred_head_10hz'][obj_mask,int(t)]=box[-1]
                            tokenized_agent['shape'][obj_mask]=box[3:6]
                            tokenized_agent["all_valid"][obj_mask]=True

        if self.input_json_path is not None :
            token_dict = self.planner.token_processor._match_agent_token(
                tokenized_agent["all_valid"][control_mask],
                tokenized_agent['pred_traj_10hz'][control_mask],
                tokenized_agent['pred_head_10hz'][control_mask],
                tokenized_agent["token_agent_shape"][control_mask],
                tokenized_agent["token_traj"][control_mask],
            )
            for key,value in token_dict.items():
                tokenized_agent[key][control_mask] = value

        pos = tokenized_agent["pred_traj_10hz"]
        heading = tokenized_agent["pred_head_10hz"]
        valid=tokenized_agent["all_valid"]

        agent_pos=pos[:,self.timestamp].cpu().numpy()
        agent_head=heading[:,self.timestamp].cpu().numpy()
        agent_valid=valid[:,self.timestamp].cpu().numpy()

        if self.GUI_DISPLAY:
            self.gui.renderQueue.put((agent_pos, agent_head, agent_type, agent_valid,self.timestamp))

        output_start=time.time()
        #rss_before = get_process_memory()
        self.dump_result(tokenized_agent)
        #print(get_self_gpu_usage())
        #print(print_cpu_usage())

        self.output_time.append(time.time()-output_start)
        #print(get_process_memory() - rss_before)

        print("time step: ",self.timestamp)

        sleep(100)
        self.capture_viewport_frame()
        self.timestamp += 1

        return True

    def run_simulation(self):
        input_dir =self.config["data_path"]
        all_scenarios=os.listdir(input_dir)
        all_scenarios.sort()
        #i=0
        for scenario in all_scenarios:
            # 记录系统层进程内存
            rss_before = get_process_memory()

            start_time = time.time()
            remove_map_object=self.config["map_object"]["remove"]
            add_map_object=self.config["map_object"]["add"]

            remove_mapid=[]
            self.route={}

            if remove_map_object is not None:

                for polyline in remove_map_object:
                    remove_mapid.append(polyline["id"])

            if add_map_object is not None:

                for polyline in add_map_object:
                    remove_mapid.append(polyline["global_id"])

            if 'desay' not in input_dir:

                dataset = tf.data.TFRecordDataset(
                    [input_dir+'/'+scenario], compression_type="", #num_parallel_reads=3
                )
                tf_data = next(iter(dataset))

                tf_data = tf_data.numpy()
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(bytes(tf_data))
                track_infos = decode_tracks_from_proto(scenario)
                map_infos = decode_map_features_from_proto(scenario.map_features,remove_mapid)
                point_cnt=len(map_infos['all_polylines'])

                if add_map_object is not None:

                    for polyline in add_map_object:
                        feature_data_type= polyline["polygon_type"]
                        cur_info = {"id": polyline["id"]}
                        cur_info["type"] = polyline["polyline_type"]

                        cur_polyline = np.stack(
                            [
                                np.array([p[0], p[1], 0, cur_info["type"], cur_info["id"]])
                                for p in polyline["points"]
                            ],
                            axis=0,
                        )
                        cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))

                        map_infos[feature_data_type].append(cur_info)
                        map_infos["all_polylines_list"].append(cur_polyline)
                        point_cnt += len(cur_polyline)

                    map_infos["all_polylines"] = np.concatenate(map_infos["all_polylines_list"], axis=0).astype(np.float32)

            else:
                with open(input_dir+'/'+scenario, "r") as f:
                    data = json.load(f)
                map_infos = decode_map_features_from_json(data['annotation'], remove_mapid,add_map_object)

                boundary_dict=map_infos["boundary_dict"]
                line_dict=map_infos["line_dict"]

                TG = TrafficGenerator(map_infos["edge_graph"], map_infos["lane_graph"],boundary_xyz=map_infos["boundary_dict"])  # 或传入你已有的 router_func

                ego_start=None
                ego_goal=None
                add_agents=self.config["agent"]["add"]

                if add_agents is not None:
                    for agent in add_agents:
                        if agent['id']==0:
                            ego_start= np.array(agent["position"])
                            if agent["goal"] is not None:
                                ego_goal= np.array(agent["goal"])
                            else:
                                ego_heading = np.array(agent["heading"])

                self.route={}

                if ego_goal is None and ego_start is None:
                    ego_edge_ids, ego_route_xyz, ego_start_xy, ego_goal_xy = TG.random_ego_edge_route(
                        seed=self.random_seed,
                        min_len_m=40.0,
                        max_len_m=3000.0,
                        sample_start_on_edge=True,  # 起点弧长随机
                        end_at_last_point=True,  # 终点为末尾 edge 的尾点
                    )
                elif ego_start is not None:
                    if ego_goal is not None:
                        ego_edge_ids, dist_m, ego_route_xyz, start_eid, goal_eid = TG._route(ego_start, ego_goal)
                    else:
                        while True:
                            try:
                                heading_goal_range = (100.0, 300.0)
                                dist = float(np.random.uniform(*heading_goal_range))
                                hd = float(ego_heading)
                                dir_vec = np.array([np.cos(hd), np.sin(hd)], float)
                                # lateral perturbation up to ±lateral_perturb_max (perpendicular to heading)
                                nrm_vec = np.array([-np.sin(hd), np.cos(hd)], float)
                                lat = float(np.random.uniform(-50, 50))
                                ego_goal = ego_start[:2] + dist * dir_vec + lat * nrm_vec

                                ego_edge_ids, dist_m, ego_route_xyz, start_eid, goal_eid = TG._route(ego_start,
                                                                                                     ego_goal)
                                break
                            except:
                                continue

                    route_xy = np.stack([ego_start, ego_goal], axis=0)#append_segment_with_step(ego_start, ego_route_xyz[:, :2], ego_goal, step=2.0)

                    self.route[0]=(torch.FloatTensor(route_xy).cuda(),13.9)

                all_agents = TG.generate_batch(
                    density01=self.config["agent_density"],
                    class_ratio=self.config["agent_class_ratio"],
                    size_tab=self.config["agent_size"],
                    ego_edge_ids=ego_edge_ids,
                    seed=self.random_seed
                )
                # add agent
                if add_agents is not None:
                    for agent in add_agents:
                        id = agent["id"]
                        type = agent["type"]
                        position = np.array(agent["position"])

                        if agent['speed'] is None:
                            speed = TG.default_speed[type]#None #
                        else:
                            speed = agent["speed"]

                        if agent["shape"] is None:
                            shape = self.config["agent_size"][type]
                        else:
                            shape = agent["shape"]

                        if agent["goal"] is not None:
                            goal = np.array(agent["goal"])
                            # path, dist_m, route_xyz, start_eid, goal_eid = TG._route(position, goal)
                            #
                            # route_xy = append_segment_with_step(position, route_xyz[:, :2], goal, step=2.0)

                            route_xy=np.stack([position, goal], axis=0)

                            self.route[id] = (torch.FloatTensor(route_xy).cuda(),speed)

                            # heading = np.arctan2(route_xy[1, 1] - route_xy[0, 1], route_xy[1, 0] - route_xy[0, 0])

                        if agent["heading"] is not None:
                            heading = agent["heading"]

                        all_agents[id]=dict(
                                cls=type,
                                size_lwh_m=shape,
                                avg_speed_mps=speed,
                                start_xyz=np.array([position[0],position[1],0]),
                                start_heading_rad=heading,
                            )

                if self.config["agent"]["remove"] is not None:
                    deleta_id = [agent["id"] for agent in self.config["agent"]["remove"]]
                    all_agents = {id: obj for id, obj in all_agents.items() if id not in deleta_id}

                agent_num=len(all_agents)#len(agents)

                track_infos = {
                    "object_id": np.arange(agent_num),
                    "object_type": np.zeros([agent_num]),
                    "states": np.zeros([agent_num,91,9]),
                    "valid": np.ones([agent_num,91]).astype(bool),
                    "role": np.zeros([agent_num,3]).astype(bool),
                }

                self.type=[]

                for j,(id,agent) in enumerate(all_agents.items()):
                    track_infos["object_id"][j]=id
                    size_lwh_m=agent["size_lwh_m"]
                    speed=agent["avg_speed_mps"]
                    heading=agent["start_heading_rad"]

                    if speed is not None:
                        velocity=np.array([np.cos(heading)*speed,np.sin(heading)*speed,0])
                        track_infos['states'][j, :, :3] = agent["start_xyz"]+velocity[None]*np.arange(-1,8.1,0.1)[:,None]
                    else:
                        track_infos['states'][j, :, :3] = agent["start_xyz"]
                        track_infos["valid"][j,:10]=False

                    track_infos['states'][j, :, 3] = size_lwh_m[0]
                    track_infos['states'][j, :, 4] = size_lwh_m[1]
                    track_infos['states'][j, :, 5] = size_lwh_m[2]
                    track_infos['states'][j, :, 6] = heading

                    self.type.append(agent["cls"])

                    if agent["cls"]=="pedestrian":
                        track_infos["object_type"][j]=1
                    elif agent["cls"]=="bicycle":
                        track_infos["object_type"][j]=2

                track_infos['role'][track_infos["object_id"]==0] = True

                track_infos["role"][:,-1]=True

                spec = StaticSpec(
                    density01=self.config["static_density"],
                    ratios=self.config["static_class_ratio"],
                    sizes_lwh_m=self.config["static_size"],
                    seed=self.random_seed
                )

                static_objs = generate_static_elements_from_raw(
                    boundary_dict=boundary_dict,
                    lane_dict=line_dict,  # using lane_line as lane dividers
                    EG=map_infos["edge_graph"],
                    lane_graph=map_infos["lane_graph"],
                    ego_edge_ids=ego_edge_ids,  # <— considered for corridor focus
                    ego_route_xyz=ego_route_xyz,  # optional
                    spec=spec
                )

                if self.light is not None:
                    for light in self.light:
                        static_objs[-1-len(static_objs)]=dict(
                            cls='light',
                            size_lwh_m=light["size"],
                            x=light["position"][0], y=light["position"][1], z=0,
                            heading_rad=light["heading"]
                        )

                if self.traffic_signs is not None:
                    for traffic_sign in self.traffic_signs:
                        static_objs[-1-len(static_objs)]=dict(
                            cls='traffic_sign',
                            size_lwh_m=traffic_sign["size"],
                            x=traffic_sign["position"][0], y=traffic_sign["position"][1], z=0,
                            heading_rad=traffic_sign["heading"]
                        )


                # add static object
                add_static = self.config["static_object"]["add"]

                if add_static is not None:
                    for static in add_static:
                        id = static['id']
                        type = static["type"]

                        if static["shape"] is None:
                            shape = self.config["static_size"][type]
                        else:
                            shape = static["shape"]

                        static_objs[id] = dict(
                            cls=type,
                            size_lwh_m=shape,
                            x=float(static["position"][0]),
                            y=float(static["position"][1]),
                            z=0,
                            heading_rad=static["heading"],
                        )

                if self.config["static_object"]["remove"] is not None:
                    deleta_id = [agent["id"] for agent in self.config["static_object"]["remove"]]
                    static_objs = {id: obj for id, obj in static_objs.items() if id not in deleta_id}

                # static_objects
                if len(static_objs):
                    static_list = []
                    static_pos, static_yaw, static_size, static_type, static_id = [], [], [], [], []

                    for id, object in static_objs.items():
                        object_type = object["cls"]

                        new_state = np.zeros([91, 9])

                        if object_type == "cone":
                            static_type.append(1)
                        elif object_type == "water_barrier":
                            static_type.append(0)
                        elif object_type == "light":
                            static_type.append(3)
                        elif object_type == "hydrant":
                            static_type.append(2)
                        elif object_type == "traffic_bollard":
                            static_type.append(4)
                        elif object_type == "stone_bollard":
                            static_type.append(5)
                        elif object_type == "traffic_sign":
                            static_type.append(6)

                        new_state[:, 0] = object["x"]
                        new_state[:, 1] = object["y"]
                        new_state[:, 3:6] = np.array(object["size_lwh_m"])[None]
                        new_state[:, 6] = object["heading_rad"]

                        static_list.append(new_state)

                        static_pos.append((object["x"], object["y"]))
                        static_yaw.append(object["heading_rad"])
                        static_size.append(object["size_lwh_m"])
                        static_id.append(id)
                        self.type.append(object_type)

                    static_pos = np.array(static_pos)
                    static_yaw = np.array(static_yaw)[:, None]
                    static_size = np.array(static_size)
                    static_type = np.array(static_type)
                    static_id = np.array(static_id)

                    new_state = np.stack(static_list)

                    new_type = np.array(static_type)
                    new_type[static_type != 0] = 1

                    track_infos["states"] = np.concatenate([track_infos["states"], new_state])
                    track_infos["object_id"] = np.concatenate([track_infos["object_id"], static_id])
                    track_infos["valid"] = np.concatenate(
                        [track_infos["valid"], np.ones([len(static_list), 91]).astype(bool)])
                    track_infos["role"] = np.concatenate(
                        [track_infos["role"], np.zeros([len(static_list), 3]).astype(bool)])
                    track_infos["object_type"] = np.concatenate([track_infos["object_type"], new_type])

                self.type=np.array(self.type)

            data_loadding_time=time.time()

            print("data load time:", data_loadding_time-start_time)

            data_loadding_memory = get_process_memory()
            print(f"data load memory  : {data_loadding_memory - rss_before:.1f} MB")
            #print(get_self_gpu_usage())
            #print(print_cpu_usage())

            map_data = get_map_features(map_infos, {})

            if 'desay' not in input_dir:
                data = preprocess_map(map_data,break_dist=3)
            else:
                data = preprocess_map(map_data,break_dist=30)

            data["agent"] = get_agent_features(
                track_infos,
                split="validation",
                num_historical_steps=11,
                num_steps=91,
            )
            data["agent"]["batch"]=torch.zeros(data["agent"]["num_nodes"]).long()
            data["pt_token"]["batch"]=torch.zeros(data["pt_token"]["num_nodes"]).long()

            data["routing"] = self.route
            data["static"] = (static_pos, static_yaw, static_size, static_type,static_id)

            self.initialize_simulation(map_data,data)

            self.agent_mask=data["agent"]["role"][:,-1]

            batch_data = HeteroData(data).cuda()
            batch_data.num_graphs=1

            tokenized_map, tokenized_agent = self.planner.token_processor(batch_data)#,extrapolate=False

            for key in ["sampled_idx","sampled_pos","sampled_heading","valid_mask","token_mask",'gt_idx']:
                pad_value=tokenized_agent[key][:,-1:].repeat(1,self.MAX_SIM_TIME//5-tokenized_agent[key].shape[1], *([1] * (tokenized_agent[key].ndim - 2)))
                tokenized_agent[key]=torch.cat([tokenized_agent[key],pad_value],dim=1)

            for key in ["pred_traj_10hz","pred_head_10hz","all_valid"]:
                pad_value=tokenized_agent[key][:,-1:].repeat(1,self.MAX_SIM_TIME+1-tokenized_agent[key].shape[1], *([1] * (tokenized_agent[key].ndim - 2)))
                tokenized_agent[key]=torch.cat([tokenized_agent[key],pad_value],dim=1)

            #tokenized_agent["vis_mask"]=tokenized_agent["type"]<5

            # route_map_index = torch.zeros([len(tokenized_agent["sampled_idx"]), 100]).to(torch.int16) - 1
            #
            # map_type = tokenized_map['type']
            # mask45 = (map_type == 4) | (map_type == 5)
            #
            # edge_xy = tokenized_map["position"][mask45].cpu().numpy()
            #
            # for id,route_xyz in data["routing"].items():
            #
            #     yaw_interp = compute_yaw_from_traj(route_xyz)
            #
            #     L_idx, R_idx, L_d, R_d = nearest_edges_biside(
            #         route_xyz[:,:2], yaw_interp, edge_xy, k=16, radius=40.0
            #     )
            #
            #     all_idx = torch.tensor(
            #         np.unique(np.concatenate([L_idx, R_idx])))  # idx4_in_45[np.unique(np.concatenate([L_idx,R_idx]))]
            #     n = min(len(all_idx), 100)
            #
            #     # import  matplotlib.pyplot as plt
            #     #
            #     # for boud in boundary_dict.values():
            #     #
            #     #    plt.scatter(boud[:,0], boud[:,1], c="green")
            #     #
            #     # edge_point=edge_xy[all_idx.numpy()]
            #     # plt.scatter(edge_point[:,0], edge_point[:,1], c="r")
            #     #
            #     # plt.show()
            #     # print(len(all_idx))
            #
            #     route_map_index[id][:n] = all_idx[:n]
            #
            # tokenized_agent["route_map_index"]=route_map_index.cuda()

            #set mean speed:
            # mean_speed=torch.zeros(len(tokenized_agent["type"])).cuda()-1
            #
            # mean_speed[0]=30
            #
            # tokenized_agent["mean_speed"]=mean_speed

            goal_pos=torch.zeros_like(tokenized_agent["sampled_pos"][:,0])
            goal_mask=torch.zeros_like(goal_pos[:,0]).to(torch.bool)

            for id,(route_xyz,speed) in self.route.items():
                idx=tokenized_agent['id']==id
                goal_pos[idx]=route_xyz[-1]
                goal_mask[idx]=True

            tokenized_agent["goal_pos"]=goal_pos
            tokenized_agent["goal_mask"]=goal_mask

            data_preproces_time=time.time()

            print("data preprocess time:", data_preproces_time-data_loadding_time)
            data_preproces_memory = get_process_memory()
            print(f"data preprocess memory  : {data_preproces_memory - data_loadding_memory:.1f} MB")
            #print(get_self_gpu_usage())
           # print(print_cpu_usage())

            map_feature = self.planner.encoder.map_encoder(tokenized_map)

            map_embedding_time=time.time()

            print("map embedding time:", map_embedding_time-data_preproces_time)
            # map_embedding_memory = get_process_memory()
            # print(f"map embedding memory  : {map_embedding_memory - data_preproces_memory:.1f} MB")
            #print(get_self_gpu_usage())
            #print(print_cpu_usage())
            while True:
                if not self.process_frame(data,map_feature, tokenized_agent):
                    break

            # print("camera_rendering_time:",np.mean(self.camera_rendering_time))
            # print("traffic_model_time:",np.mean(self.traffic_model_time))
            # print("output_time:",np.mean(self.output_time))

            self.cleanup()

            return

    def capture_viewport_frame(self):
        if not self.recording or self.video_writer is None:
            return

        if self.timestamp==self.initial_step:
            sleep(0.1)

        import mss

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


    def dump_result(self,tokenized_agent):

        result={}
        no_ego = tokenized_agent["all_valid"][:,self.timestamp]
        tracking_id = tokenized_agent["id"]

        ego_mask=tokenized_agent["ego_mask"]

        no_ego[ego_mask]=False #no ego export

        pos_global = tokenized_agent["pred_traj_10hz"][:,self.timestamp]
        prev_pos =  tokenized_agent["pred_traj_10hz"][:,self.timestamp-1]
        head_global = tokenized_agent["pred_head_10hz"][:,self.timestamp]
        shape=tokenized_agent["shape"]

        ego_pos=pos_global[ego_mask]
        ego_heading=head_global[ego_mask]

        pos_global=pos_global[no_ego]
        head_global=head_global[no_ego]
        prev_pos=prev_pos[no_ego]
        shape=shape[no_ego]
        tracking_id=tracking_id[no_ego]

        pos_local, head_local=transform_to_local(pos_global[None],head_global[None],ego_pos,ego_heading)
        prev_pos_local=transform_to_local(prev_pos[None],None,ego_pos,ego_heading)[0]

        pos_local=pos_local[0]
        head_local=head_local[0]
        prev_pos_local=prev_pos_local[0]

        head_local=wrap_angle(head_local)

        velocity=(pos_local-prev_pos_local)/0.1

        boxes=torch.zeros([len(pos_local),7]).cuda()#[x, y, z, l, w, h, yaw]

        boxes[:,:2]=pos_local
        boxes[:,2]=shape[:,2]/2-self.lidar_height  #half height
        boxes[:,3:6]=shape
        boxes[:,6]=head_local

        # from collections import defaultdict
        #
        # groups = defaultdict(list)  # (H,W) -> [cam_idx]
        # for i, key in enumerate(self.lidar2img.keys()):
        #     groups[self.image_size[key]].append(i)
        #
        # cams_all = [torch.as_tensor(self.lidar2img[k], dtype=torch.float32, device=boxes.device)
        #             for k in self.lidar2img.keys()]
        # cams_all = torch.stack(cams_all, 0)  # (K,4,4)
        #
        # visible_accum = torch.zeros((1, boxes.shape[0]), dtype=torch.bool, device=boxes.device)
        # for (H, W), idxs in groups.items():
        #     cams_grp = cams_all[idxs]  # (Kg,4,4)
        #     vis_grp = check_occlusion_multi_cam(boxes[None], cams_grp, (H, W))  # (1,N)
        #     visible_accum |= vis_grp
        #
        # visible1 = visible_accum.to(torch.int).cpu().numpy()[0]  # (N,)

        # Prepare batched cameras (same resolution case)
        cams = torch.stack(
            [torch.as_tensor(self.lidar2img[k], dtype=torch.float32, device=boxes.device)
             for k in self.lidar2img.keys()],
            dim=0
        )  # (K,4,4)

        # If every camera has the same (H,W):
       # Hs, Ws = zip(*self.image_size.values())
        H, W = self.image_size["CAM_FRONT"]
        visible = check_occlusion_multi_cam(
            boxes[None], cams, (H, W), tie_break_first=False  # set True if exact tie behavior matters
        )  # (1,N) bool
        visible = visible.to(torch.int).cpu().numpy()[0]

        boxes=boxes.cpu().numpy()
        tracking_id=tracking_id.cpu().numpy()

        labels=np.array(self.type[no_ego.cpu().numpy()])

        # Read JSON from file
        if self.timestamp>self.initial_step:
            with open(self.output_json_path, 'r') as f:
                result = json.load(f)

        if self.config['export_mode']=="agents":
            export_mask=tracking_id>0
        else:
            export_mask=np.ones_like(visible).astype(np.bool)

        t=str(self.timestamp)

        result[t]={}
        result[t]["bboxes"]=boxes[export_mask].tolist()
        result[t]["labels"]=labels[export_mask].tolist()
        result[t]["tracking_id"]=tracking_id[export_mask].tolist()
        result[t]["velocity"]=velocity[export_mask].tolist()
        result[t]["occluded"]=visible[export_mask].tolist()

        with open(self.output_json_path, "w") as f:
            json.dump(result, f, indent=2)

        # Read JSON from file
        if self.timestamp>self.initial_step:
            with open(self.output_ego_json_path, 'r') as f:
                ego_result = json.load(f)
        else:
            ego_result=[[]]

        # all_valid[tracking_id<0]=False
        result={"POSE":{}}
        result["POSE"]["timestamp"]=str(self.timestamp)

        z=0 #self.lidar_height
        x=ego_pos[0][0].cpu().numpy()
        y=ego_pos[0][1].cpu().numpy()
        yaw=ego_heading[0].cpu().numpy()
        cos=np.cos(yaw)
        sin=np.sin(yaw)

        matrix=np.array([[cos,-sin,0,x],
                         [sin,cos,0,y],
                         [0, 0, 1, z],
                         [0, 0, 0, 1]
                         ])

        result["POSE"]["data"]=matrix.tolist()

        ego_result[0].append(result)

        with open(self.output_ego_json_path, "w") as f:
            json.dump(ego_result, f, indent=2)


    def cleanup(self):

        print("Simulation ends")

        if self.GUI_DISPLAY:
            self.gui.terminate()
            self.gui.join()
        if self.recording:
            self.video_writer.release()

    def setup_planner(self,cfg):
        self.planner = SMART_IQ(cfg.model.model_config)

        # if torch.cuda.is_available():
        #     state_dict = torch.load(self.config["planner_path"],weights_only=False)["state_dict"]
        # else:
        #     state_dict = torch.load(self.config["planner_path"], map_location=torch.device("cpu"),weights_only=False)["state_dict"]
        #
        # self.planner.load_state_dict(state_dict)#,strict=False
        self.planner.cuda()
        self.planner.eval()


@hydra.main(config_path="../configs/", config_name="run.yaml", version_base=None)
def main(model_cfg):

    default_config='./config.yaml'

    simulator_cfg = OmegaConf.load(default_config)

    custom_config_path=model_cfg.cfg


    # Merge if provided
    if custom_config_path is not None:
        print(f"Loading custom config from: {custom_config_path}")
        override_cfg = OmegaConf.load(custom_config_path)
        simulator_cfg = OmegaConf.merge(simulator_cfg, override_cfg)


    sim_manager = SimulationManager(model_cfg, simulator_cfg)
    sim_manager.run_simulation()


if __name__ == '__main__':
    main()
