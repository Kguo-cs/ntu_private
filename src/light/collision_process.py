import multiprocessing
import os
import pickle
from tqdm import tqdm
import torch
import datetime
from torch_geometric.data import HeteroData
import numpy as np
from torch_geometric.data import HeteroData
torch.set_float32_matmul_precision("high")
from src.smart.tokens.my_token_processor import TokenProcessor
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)
import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import shapely
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon as PolygonPatch

data_directory = "/home/ke/code/catk/src/waymo_data/full/training_inter10"
output_path = "/home/ke/code/catk/src/waymo_data/full/training_inter10_col/"

os.makedirs(output_path, exist_ok=True)

token_processor = TokenProcessor(
    map_token_file="map_traj_token5.pkl",
    agent_token_file="agent_vocab_555_s2.pkl",
    map_token_sampling={"num_k": 1, "temp": 1.0},
    agent_token_sampling={"num_k": 1, "temp": 1.0}
)
token_processor.eval()

files = os.listdir(data_directory)

for filename in tqdm(files):
    input_path = os.path.join(data_directory, filename)
    with open(input_path, "rb") as f:
        data = pickle.load(f)


    agent = data["tokenized_agent"]

    # token_agent_shape, token_traj_all, token_traj = token_processor._get_agent_shape_and_token_traj(
    #     agent["type"]
    # )

    sampled_pos=agent["sampled_pos"]
    sampled_heading=agent["sampled_heading"]
    agent_shape=agent["shape"]
    valid=agent["valid_mask"]
    agent_shape = agent_shape[:, None].repeat(1, sampled_pos.shape[1], 1)  # [:,:,None][:,0]

    polygon=cal_polygon_contour(sampled_pos,sampled_heading,agent_shape)
    node_capacity=1000

    mask=torch.zeros_like(valid)

    for t in range(polygon.shape[1]):
        geometries=polygon[:,t][valid[:,t]]

        _geometries = [Polygon(geometry) for geometry in geometries]
        _str_tree = STRtree(_geometries, node_capacity)
        fig, ax = plt.subplots()

        for i,ego_polygons in enumerate(_geometries):

            intersecting = _str_tree.query(ego_polygons, predicate="intersects")

            if len(intersecting)>1:
                k=np.arange(len(valid))[valid[:,t]][i]
                mask[k,t]=True

        data["tokenized_agent"]["col_mask"]=mask

        for geom in geometries:
            polygon1 = PolygonPatch(geom, closed=True, edgecolor='blue', facecolor='lightblue')
            ax.add_patch(polygon1)

        ax.set_xlim(580, 680)
        ax.set_ylim(11040, 11140)

        plt.show()
        print(len(intersecting))



    # sampled_idx=agent["sampled_idx"].long()bc56_light5_100_sharet_temp_poshead
    #
    # traj=token_traj_all[np.arange(len(sampled_idx))[:,None],sampled_idx]

    # cos, sin = head_now.cos(), head_now.sin()
    # rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    # rot_mat[:, 0, 0] = cos
    # rot_mat[:, 0, 1] = sin
    # rot_mat[:, 1, 0] = -sin
    # rot_mat[:, 1, 1] = cos
    #
    # pos_global = torch.bmm(pos_local, rot_mat)  # [n_agent, n_step, 2]*[n_agent, 2, 2]
    # pos_global = pos_global + pos_now.unsqueeze(1)
    # if head_local is None:
    #     head_global = None
    # else:
    #     head_global = head_local + head_now.unsqueeze(1)



    #traj1=traj.reshape(len(traj)*18,-1,2)

    # prev_pos=torch.concat([pos[:,:1],sampled_pos[:,:-1]],dim=1).reshape(-1,2)
    # prev_head=torch.concat([heading[:,:1],sampled_heading[:,:-1]],dim=1).reshape(-1)
    # #valid 0 & 5
    # token_world_gt = transform_to_global(
    #     pos_local=traj1,  # [n_agent, n_token*4, 2]
    #     head_local=None,
    #     pos_now=prev_pos,  # [n_agent, 2]
    #     head_now=prev_head,  # [n_agent]
    # )[0].view(traj.shape)

    # output_file = output_path + filename
    #
    # with open(output_file, "wb") as f:
    #     pickle.dump(data, f)
    #


