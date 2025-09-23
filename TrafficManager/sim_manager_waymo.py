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
seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
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

from src.smart.model.smart import SMART

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
from pynvml import *

from desay_utils.check_oclluded import check_occlusion_fully_batched
from desay_utils.decay_data_process import decode_map_features_from_json
from desay_utils.idm_policy import idm_planner
from desay_utils.random_trip import TrafficGenerator
from collections import Counter

def print_cpu_usage(interval=1.0):
    pid = os.getpid()
    process = psutil.Process(pid)

    # 初次调用时统计间隔 CPU 时间，第二次才有结果
    process.cpu_percent(interval=None)  # 清除旧状态
    time.sleep(interval)
    cpu_usage = process.cpu_percent(interval=None)

    print(f"🧠 当前进程 CPU 占用率：{cpu_usage:.2f}%")

def get_self_gpu_usage():
    nvmlInit()
    pid = os.getpid()
    device_count = nvmlDeviceGetCount()

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        procs = nvmlDeviceGetComputeRunningProcesses(handle)

        for p in procs:
            if p.pid == pid:
                meminfo = nvmlDeviceGetMemoryInfo(handle)
                util = nvmlDeviceGetUtilizationRates(handle)
                print(f"[GPU:{i}] PID={pid} 使用显存: {p.usedGpuMemory / 1024**2:.1f} MiB / {meminfo.total / 1024**2:.1f} MiB")
                print(f"[GPU:{i}] GPU 核心利用率: {util.gpu}%")
                break

    nvmlShutdown()


def get_process_memory():
    """返回当前进程的 RSS（真实占用内存，单位 MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

class SimulationManager:
    def __init__(self, model_cfg,config: str) -> None:
        self.config = config
        self.setup_planner(model_cfg)
        self.GUI_DISPLAY = self.config["gui_display"]

        data_root = os.path.dirname(os.path.abspath(__file__))

        self.DATA_TEMPLATE_PATH = os.path.join(
            data_root, self.config["data"]["template_path"]
        )
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


        self.camera_rendering_time=[]
        self.traffic_model_time=[]
        self.output_time=[]

        self.random_seed=int(self.config["random_seed"])

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def initialize_simulation(self,map_data,data):
        # Initialising models, planners, maps etc
        light = self.config["traffic_lights"]

        self.gui = GUI(map_data,data,light,self.initial_step)
        if self.GUI_DISPLAY:
            self.gui.start()

        self.data_template = torch.load(self.DATA_TEMPLATE_PATH,weights_only=False)

        self.timestamp = self.initial_step
        self.MAX_SIM_TIME = self.config["max_sim_time"]

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

        if self.timestamp % 5 == 0:
            if self.GUI_DISPLAY:
                camera_rendering_start = time.time()
                #rss_before = get_process_memory()

                agent_pos = tokenized_agent['sampled_pos'][:,self.timestamp//5-1].cpu().numpy()
                agent_heading=tokenized_agent['sampled_heading'][:,self.timestamp//5-1].cpu().numpy()

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

            # with torch.no_grad():
            #     pred_dict = self.planner.encoder.agent_encoder.inference( tokenized_agent, map_feature ,step_current_10hz=self.timestamp,n_step_future_10hz=5 )
            #
            # for key in ["sampled_idx","sampled_pos","sampled_heading","valid_mask"]:
            #     pred_value=pred_dict[key]
            #     tokenized_agent[key][self.control_mask,:pred_value.shape[1]] = pred_value[self.control_mask]
            #
            # for key in ["pred_traj_10hz","pred_head_10hz"]:
            #     tokenized_agent[key][self.control_mask,self.timestamp+1:self.timestamp+6] = pred_dict[key][self.control_mask]

            tokenized_agent["all_valid"][self.control_mask, self.timestamp + 1:self.timestamp + 6] = True

            for id, route in self.route.items():
                idx = torch.where(tokenized_agent["id"] == id)[0]

                all_pos = tokenized_agent["pred_traj_10hz"][:, self.timestamp]

                all_heading = tokenized_agent["pred_head_10hz"][:, self.timestamp]

                all_shape = tokenized_agent["shape"][:, :2]  # length, width

                # route [n,2]
                prev_pos = tokenized_agent["pred_traj_10hz"][:, self.timestamp - 1]

                all_velocity = (all_pos - prev_pos) / 0.1

                new_pos, new_heading = idm_planner(route, idx, all_pos, all_heading, all_velocity, all_shape,
                                                   desired_speed=20)  # plan 0.5 second

                tokenized_agent["pred_traj_10hz"][idx, self.timestamp + 1:self.timestamp + 6]=new_pos
                tokenized_agent["pred_head_10hz"][idx, self.timestamp + 1:self.timestamp + 6]=new_heading

                token_dict = self.planner.token_processor._match_agent_token(
                    tokenized_agent["all_valid"],
                    tokenized_agent['pred_traj_10hz'],
                    tokenized_agent['pred_head_10hz'],
                    tokenized_agent["token_agent_shape"],
                    tokenized_agent["token_traj"],
                )

                tokenized_agent.update(token_dict)

            self.traffic_model_time.append(time.time()-traffic_model_start)
           # print(get_process_memory() - rss_before)
            #print(get_self_gpu_usage())
            #print(print_cpu_usage())

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
                            tokenized_agent['pred_traj_10hz'][obj_mask,int(t)]=box[:2]
                            tokenized_agent['pred_head_10hz'][obj_mask,int(t)]=box[-1]
                            tokenized_agent['shape'][obj_mask]=box[3:6]
                            tokenized_agent["all_valid"][obj_mask]=True

                    token_dict =self.planner.token_processor._match_agent_token(
                                                                             tokenized_agent["all_valid"],
                                                                             tokenized_agent['pred_traj_10hz'],
                                                                             tokenized_agent['pred_head_10hz'],
                                                                             tokenized_agent["token_agent_shape"],
                                                                             tokenized_agent["token_traj"],
                                                                             )
                    tokenized_agent.update(token_dict)

        pos = tokenized_agent["pred_traj_10hz"]
        heading = tokenized_agent["pred_head_10hz"]

        light_idx = []#tokenized_agent["light_idx"][:,(self.timestamp-5)//5].cpu().numpy()
        agent_pos=pos[:,self.timestamp].cpu().numpy()
        agent_head=heading[:,self.timestamp].cpu().numpy()

        if self.GUI_DISPLAY:
            self.gui.renderQueue.put((agent_pos, agent_head, agent_type, light_idx,self.timestamp))

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
                    remove_mapid.append(polyline["id"])

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

            else:
                with open(input_dir+'/'+scenario, "r") as f:
                    data = json.load(f)
                map_infos = decode_map_features_from_json(data['annotation'], remove_mapid)

                # agent_num=1
                #
                # route=generate_random_edge_trips(map_infos["edge_graph"],map_infos["lane_graph"],n_trips=agent_num,seed=self.random_seed)
                TG = TrafficGenerator(map_infos["edge_graph"], map_infos["lane_graph"],boundary_xyz=map_infos["boundary_xyz"])  # 或传入你已有的 router_func

                # 假设已有 TG = TrafficGenerator(EG, G_lane)
                ego_edge_ids, ego_route_xyz, ego_start_xy, ego_goal_xy = TG.random_ego_edge_route(
                    seed=self.random_seed,
                    min_len_m=40.0,
                    max_len_m=3000.0,
                    sample_start_on_edge=True,  # 起点弧长随机
                    end_at_last_point=True  # 终点为末尾 edge 的尾点
                )
                agents = TG.generate_batch(
                    density01=1,
                    class_ratio={"pedestrian": 1, "car": 8, "truck": 2, "bicycle": 1},
                    ego_edge_ids=ego_edge_ids,
                    seed=self.random_seed,
                    lift_to_lane_ids=True
                )

                counts = Counter(a["cls"] for a in agents)
                print("Agent counts by type:")
                for cls, n in counts.items():
                    print(f"  {cls}: {n}")
                    # print(agents[0].agent_id, agents[0].cls, agents[0].avg_speed_mps, len(agents[0].edge_ids))

                agent_num=1+len(agents)#len(agents)

                track_infos = {
                    "object_id": np.arange(agent_num),
                    "object_type": np.zeros([agent_num]),
                    "states": np.zeros([agent_num,91,9]),
                    "valid": np.ones([agent_num,91]).astype(bool),
                    "role": np.zeros([agent_num,3]).astype(bool),
                }

                # for i,route_i in enumerate(route):

                route=ego_route_xyz[:,:2]

                track_infos['states'][0,:,:3]=ego_route_xyz[:1]
                track_infos['states'][0,:,3]=4.8
                track_infos['states'][0,:,4]=1.8
                track_infos['states'][0,:,5]=1.6
                track_infos['states'][0,:,6]=np.arctan2(route[1,1]-route[0,1],route[1,0]-route[0,0])
                track_infos['role'][0]=1

                self.route[0]=torch.FloatTensor(route).cuda()

                for i,agent in enumerate(agents):
                    j=i+1
                    size_lwh_m=agent["size_lwh_m"]
                    speed=agent["avg_speed_mps"]
                    route=agent["route_xyz"]
                    heading=agent["start_heading_rad"]

                    velocity=np.array([np.cos(heading)*speed,np.sin(heading)*speed,0])
                    track_infos['states'][j, :, :3] = agent["start_xyz"]+velocity[None]*np.arange(-1,8.1,0.1)[:,None]
                    track_infos['states'][j, :, 3] = size_lwh_m[0]
                    track_infos['states'][j, :, 4] = size_lwh_m[1]
                    track_infos['states'][j, :, 5] = size_lwh_m[2]
                    track_infos['states'][j, :, 6] = heading

                    if agent["cls"]=="pedestrian":
                        track_infos["object_type"][j]=1
                    elif agent["cls"]=="bicycle":
                        track_infos["object_type"][j]=2
                    self.route[j]=torch.FloatTensor(route[:,:2]).cuda()

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

            # dynamic_map_infos = decode_dynamic_map_states_from_proto(
            #     scenario.dynamic_map_states
            # )
            # tf_lights = process_dynamic_map(dynamic_map_infos)
            # tf_current_light = tf_lights.loc[tf_lights["time_step"] == current_time_index]
            tf_current_light={}

            current_time_index = 10 #scenario.current_time_index

            data_loadding_time=time.time()

            print("data load time:", data_loadding_time-start_time)

            data_loadding_memory = get_process_memory()
            print(f"data load memory  : {data_loadding_memory - rss_before:.1f} MB")
            #print(get_self_gpu_usage())
            #print(print_cpu_usage())

            map_data = get_map_features(map_infos, tf_current_light)

            if 'desay' not in input_dir:
                data = preprocess_map(map_data,break_dist=3)
            else:
                data = preprocess_map(map_data,break_dist=30)

            #add agent
            add_agents=self.config["agent"]["add"]

            # delete agent
            mask = np.ones(len(track_infos["object_id"])).astype(bool)

            if self.config["agent"]["remove"] is not None:
                for agent in self.config["agent"]["remove"]:
                    mask[track_infos["object_id"] == agent["id"]] = False

            if add_agents is not None:
                for agent in add_agents:
                    mask[track_infos["object_id"] == agent["id"]] = False

                add_agent_num=len(add_agents)
            else:
                add_agent_num=0

            track_infos["object_id"] = track_infos["object_id"][mask]
            track_infos["object_type"] = track_infos["object_type"][mask]
            track_infos["states"] = track_infos["states"][mask]
            track_infos["valid"] = track_infos["valid"][mask]
            track_infos["role"] = track_infos["role"][mask]

            new_state=np.zeros([add_agent_num,91,9])
            agent_type=[]
            object_id=[]

            for i in range(add_agent_num):
                agent=add_agents[i]
                pos=np.array(agent["position"])[None]+np.array(agent["velocity"])[None]*np.arange(-1,8.1,0.1)[:,None]

                new_state[i, :, :2] = pos
                new_state[i, :, 3:6] =np.array( agent["shape"])[None]
                new_state[i, :, 6] = np.arctan2(agent["velocity"][1],agent["velocity"][0])
                new_state[i,:, 7:9] =np.array(agent["velocity"])[None]

                agent_type.append(agent["type"])
                object_id.append(agent["id"])

            track_infos["object_type"]=np.concatenate([track_infos["object_type"],np.array(agent_type)])
            track_infos["states"]=np.concatenate([track_infos["states"],new_state])
            track_infos["object_id"]=np.concatenate([track_infos["object_id"],np.array(object_id)])
            track_infos["valid"]=np.concatenate([track_infos["valid"],np.ones([add_agent_num,91]).astype(bool)])
            track_infos["role"]=np.concatenate([track_infos["role"],np.zeros([add_agent_num,3]).astype(bool)])

            track_infos["role"][:,-1]=True#control

            # add static object
            add_static = self.config["static_object"]["add"]

            if add_static is not None:
                add_static_num=len(add_static)
                new_state=np.zeros([add_static_num,91,9])

                for i in range(add_static_num):
                    static=add_static[i]
                    new_state[i, :, :2] = np.array(static["position"])[None]
                    new_state[i, :, 3:6] =np.array( static["shape"])[None]
                    new_state[i, :, 6] = static["heading"]

                track_infos["states"]=np.concatenate([track_infos["states"],new_state])

                track_infos["object_id"]=np.concatenate([track_infos["object_id"],-1-np.arange(add_static_num)])
                track_infos["valid"]=np.concatenate([track_infos["valid"],np.ones([add_static_num,91]).astype(bool)])
                track_infos["role"]=np.concatenate([track_infos["role"],np.zeros([add_static_num,3]).astype(bool)])
                track_infos["object_type"]=np.concatenate([track_infos["object_type"],np.zeros([add_static_num])])


            if self.config["agent"]["stop"] is not None:
                for agent in self.config["agent"]["stop"]:
                    id=agent["id"]
                    mask=track_infos["object_id"]==id
                    track_infos["valid"][mask]=True
                    track_infos["states"][mask]=track_infos["states"][mask,10:11]
                    track_infos["role"][mask,-1]=False

            if self.config["agent"]["recording"] is not None:
                for agent in self.config["agent"]["recording"]:
                    id=agent["id"]
                    mask=track_infos["object_id"]==id
                    track_infos["role"][mask,-1]=False


            data["agent"] = get_agent_features(
                track_infos,
                split="validation",
                num_historical_steps=current_time_index + 1,
                num_steps=91,
            )
            data["agent"]["batch"]=torch.zeros(data["agent"]["num_nodes"]).long()
            data["pt_token"]["batch"]=torch.zeros(data["pt_token"]["num_nodes"]).long()


            # if 'desay' in input_dir:
            #     route=generate_random_edge_trips(map_infos["edge_graph"],map_infos["lane_graph"],n_trips=1,seed=self.random_seed)
            #
            #     #current_pos=route[0].start_xyz[:,:2]#np.array([2,20])
            #    # goal_pos=route[0].goal_xyz[:,:2]#np.array([-200,400])
            #
            #     control_id=0
            #
            #     #edge_route=route_on_edge_graph(map_infos["edge_graph"],map_infos["lane_graph"],current_pos,goal_pos)
            #
            #
            #     route=route[0].route_xyz[:,:2]
            #
            #     print(route)
            #
            #     #print(1)
            #
            #     # data["light"] = process_light(map_infos, tf_lights, tf_current_light)
            #     # data["light"]["batch"]=torch.zeros(data["light"]["num_nodes"]).long()
            #     # #
            #   #  for lane in map_infos["centerline_list"]:
            #    #     plt.plot(lane[:,0],lane[:,1])
            #
            #     # plt.show()
            #    #  control_id=1010
            #    #  current_pos=data["agent"]['position'][data["agent"]['id']==control_id][0,10,:2]
            #    #
            #    # # goal_pos=np.array([370,6325])
            #    #  goal_pos=np.array([355,6355])
            #    #
            #    #  route=route_from_centerlines(map_infos['centerline_list'],current_pos,goal_pos)
            #    #
            #    #  route=np.array(route)

            # self.route={}
            # self.route[control_id]=torch.FloatTensor(route).cuda()
            #plt.plot(route[:,0],route[:,1],linewidth=3,color='r')
           # plt.show()

            self.initialize_simulation(map_data,data)

            self.control_mask=data["agent"]["role"][:,-1]

            batch_data = HeteroData(data).cuda()
            batch_data.num_graphs=1


            tokenized_map, tokenized_agent = self.planner.token_processor(batch_data)

            for key in ["sampled_idx","sampled_pos","sampled_heading","valid_mask"]:
                pad_value=tokenized_agent[key][:,-1:].repeat(1,self.MAX_SIM_TIME//5-tokenized_agent[key].shape[1], *([1] * (tokenized_agent[key].ndim - 2)))
                tokenized_agent[key]=torch.cat([tokenized_agent[key],pad_value],dim=1)

            for key in ["pred_traj_10hz","pred_head_10hz","all_valid"]:
                pad_value=tokenized_agent[key][:,-1:].repeat(1,self.MAX_SIM_TIME+1-tokenized_agent[key].shape[1], *([1] * (tokenized_agent[key].ndim - 2)))
                tokenized_agent[key]=torch.cat([tokenized_agent[key],pad_value],dim=1)


            data_preproces_time=time.time()

            print("data preprocess time:", data_preproces_time-data_loadding_time)
            data_preproces_memory = get_process_memory()
            print(f"data preprocess memory  : {data_preproces_memory - data_loadding_memory:.1f} MB")
            #print(get_self_gpu_usage())
           # print(print_cpu_usage())

            map_feature = self.planner.encoder.map_encoder(tokenized_map)

            map_embedding_time=time.time()

            print("map embedding time:", map_embedding_time-data_preproces_time)
            map_embedding_memory = get_process_memory()
            print(f"map embedding memory  : {map_embedding_memory - data_preproces_memory:.1f} MB")
            #print(get_self_gpu_usage())
            #print(print_cpu_usage())

            # self.control_mask = tokenized_agent["type"]<3
            #
            # tokenized_agent["type"][tokenized_agent["type"]==3]=0

            while True:
                if not self.process_frame(data,map_feature, tokenized_agent):
                    break

            print("camera_rendering_time:",np.mean(self.camera_rendering_time))
            print("traffic_model_time:",np.mean(self.traffic_model_time))
            print("output_time:",np.mean(self.output_time))

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
        all_valid = tokenized_agent["all_valid"][:,self.timestamp]
        tracking_id = tokenized_agent["id"]

        ego_mask=tokenized_agent["ego_mask"]

        all_valid[ego_mask]=False
        all_valid[tracking_id<0]=False

        pos_global = tokenized_agent["pred_traj_10hz"][:,self.timestamp]
        prev_pos =  tokenized_agent["pred_traj_10hz"][:,self.timestamp-1]
        head_global = tokenized_agent["pred_head_10hz"][:,self.timestamp]
        shape=tokenized_agent["shape"]

        ego_pos=pos_global[ego_mask]
        ego_heading=head_global[ego_mask]

        pos_global=pos_global[all_valid]
        head_global=head_global[all_valid]
        prev_pos=prev_pos[all_valid]
        shape=shape[all_valid]


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

        visible_list=[]

        for key in self.lidar2img.keys():
            lidar2img=torch.FloatTensor(self.lidar2img[key]).cuda()

            visible=check_occlusion_fully_batched(boxes[None],lidar2img[None],image_size=self.image_size[key])#width, height

            visible_list.append(visible)

        visible=torch.cat(visible_list).any(dim=0).to(torch.int).cpu().numpy()

        boxes=boxes.cpu().numpy()

        type=tokenized_agent["type"].cpu().numpy()
        tracking_id=tracking_id.cpu().numpy()

        labels=np.array(["car","pedestrian","bicycle"])[type]#,"traffic_cone"

        # Read JSON from file
        if self.timestamp>self.initial_step:
            with open(self.output_json_path, 'r') as f:
                result = json.load(f)

        t=str(self.timestamp)

        result[t]={}
        result[t]["bboxes"]=boxes.tolist()
        result[t]["labels"]=labels.tolist()
        result[t]["tracking_id"]=tracking_id.tolist()
        result[t]["velocity"]=velocity.tolist()
        result[t]["occluded"]=visible.tolist()

        with open(self.output_json_path, "w") as f:
            json.dump(result, f, indent=2)

    def cleanup(self):

        print("Simulation ends")

        if self.GUI_DISPLAY:
            self.gui.terminate()
            self.gui.join()
        if self.recording:
            self.video_writer.release()

    def setup_planner(self,cfg):
        self.planner = SMART(cfg.model.model_config)

        # if torch.cuda.is_available():
        #     state_dict = torch.load(self.config["planner_path"],weights_only=False)["state_dict"]
        # else:
        #     state_dict = torch.load(self.config["planner_path"], map_location=torch.device("cpu"),weights_only=False)["state_dict"]
        #
        # self.planner.load_state_dict(state_dict,strict=False)
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
