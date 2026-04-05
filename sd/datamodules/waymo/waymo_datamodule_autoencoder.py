import pytorch_lightning as pl 

from sd.datasets.waymo.dataset_autoencoder_waymo import WaymoDatasetAutoEncoder
from torch_geometric.loader import DataLoader
import os
from sd.utils.data_container import ScenarioDreamerData
import glob
import torch
from sd.cfgs.config import PROPORTION_NOCTURNE_COMPATIBLE, NON_PARTITIONED, NOCTURNE_COMPATIBLE
from sd.utils.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite

# this is so that CPUs are not suboptimally utilized
def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count())) 

class WaymoDataModuleAutoEncoder(pl.LightningDataModule):

    def __init__(self,
                 train_batch_size,
                 val_batch_size,
                 num_workers,
                 pin_memory,
                 persistent_workers,
                 dataset_cfg):
        super(WaymoDataModuleAutoEncoder, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size 
        self.num_workers = num_workers
        self.pin_memory = pin_memory 
        self.persistent_workers = persistent_workers
        self.cfg_dataset = dataset_cfg

        self.dataset_name = self.cfg_dataset.dataset_path
        self.init_prob_matrix = torch.load(os.environ["PROJECT_ROOT"]+'/metadata/initial_prob_matrix_waymo.pt')

    def _initialize_pyg_dset(self, mode, num_samples, batch_size, conditioning_path=None,
                             nocturne_compatible_only=False):
        """ Initialize a PyTorch Geometric dataset with the appropriate metadata for the given generation mode."""
        data_list = []
        map_id_counter = 0

        conditioning_files = None
        if mode == 'lane_conditioned':
            assert conditioning_path is not None, "conditioning_path must be provided for lane conditioned agent generation"
            # only load non-partitioned scenes
            if 'waymo' in self.dataset_name:
                conditioning_files = sorted(glob.glob(conditioning_path + "/*-of-*_*_0_*.pkl"))
            else:
                conditioning_files = sorted(glob.glob(conditioning_path + "/*_0.pkl"))
            conditioning_files = conditioning_files[:num_samples]
        elif mode == 'inpainting':
            assert conditioning_path is not None, "conditioning_path must be provided for inpainting generation"
            conditioning_files = sorted(glob.glob(conditioning_path + "/*_*.pkl"))
            conditioning_files = conditioning_files[:num_samples]

        for i in range(num_samples):
            d = ScenarioDreamerData()

            if mode == 'initial_scene':
                if 'waymo' in self.dataset_name:
                    if nocturne_compatible_only:
                        map_id = torch.tensor(NOCTURNE_COMPATIBLE)
                    else:
                        map_id = torch.multinomial(
                            torch.tensor([1 - PROPORTION_NOCTURNE_COMPATIBLE, PROPORTION_NOCTURNE_COMPATIBLE]), 1)
                else:
                    map_id = map_id_counter
                    map_id_counter += 1
                    map_id_counter = map_id_counter % self.cfg_dataset.num_map_ids

                lane_agent_probs = self.init_prob_matrix[map_id].reshape(1, -1)
                folded_num_lanes_agents = torch.multinomial(lane_agent_probs, 1).squeeze(-1)
                # +1 because there is an index for "no agents" and "no lanes"
                num_lanes = (folded_num_lanes_agents // (self.cfg_dataset.max_num_agents + 1)).item()
                num_agents = (folded_num_lanes_agents % (self.cfg_dataset.max_num_agents + 1)).item()

                assert num_lanes > 0 and num_agents > 0, "Generating scene with either no lanes or no agents"

                lg_type = NON_PARTITIONED  # as we are generating initial scenes

                d['map_id'] = int(map_id)
                d['lg_type'] = int(lg_type)
                d['num_lanes'] = int(num_lanes)
                d['num_agents'] = int(num_agents)
                d['lane'].x = torch.empty((num_lanes, 44))
                d['agent'].x = torch.empty((num_agents, 10))
                d['lane', 'to', 'lane'].edge_index = get_edge_index_complete_graph(num_lanes)
                d['agent', 'to', 'agent'].edge_index = get_edge_index_complete_graph(num_agents)
                d['lane', 'to', 'agent'].edge_index = get_edge_index_bipartite(num_lanes, num_agents)

                data_list.append(d)

            elif mode == 'inpainting':
                conditioning_file = conditioning_files[i]
                with open(os.path.join(conditioning_path, conditioning_file), 'rb') as f:
                    cond_d = pickle.load(f)

                if 'route' in cond_d:
                    route = cond_d['route']
                    center = route[-1]
                    _, yaw = estimate_heading(route)
                else:
                    route, found_route = sample_route(cond_d, dataset=self.cfg.dataset_name)
                    if found_route:
                        center = route[-1]
                        _, yaw = estimate_heading(route)
                    else:
                        center, yaw = get_default_route_center_yaw(dataset=self.cfg.dataset_name)

                # normalize to endpoint of route
                normalize_dict = {
                    'center': center,
                    'yaw': yaw
                }

                d = normalize_and_crop_scene(cond_d, d, normalize_dict, self.cfg_dataset, self.cfg.dataset_name)
                data_list.append(d)

            elif mode == 'lane_conditioned':
                # process conditioning data similar to ldm dataloader
                conditioning_file = conditioning_files[i]
                with open(os.path.join(conditioning_path, conditioning_file), 'rb') as f:
                    cond_d = pickle.load(f)

                agent_states = cond_d['agent_states']
                road_points = cond_d['road_points']
                lane_mu = cond_d['lane_mu']
                agent_mu = cond_d['agent_mu']
                lane_log_var = cond_d['lane_log_var']
                agent_log_var = cond_d['agent_log_var']
                edge_index_lane_to_lane = cond_d['edge_index_lane_to_lane']
                edge_index_lane_to_agent = cond_d['edge_index_lane_to_agent']
                edge_index_agent_to_agent = cond_d['edge_index_agent_to_agent']
                scene_type = cond_d['scene_type']
                if self.cfg.dataset_name == 'nuplan':
                    map_id = cond_d['map_id']
                else:
                    map_id = cond_d['nocturne_compatible']
                num_lanes = lane_mu.shape[0]
                num_agents = agent_mu.shape[0]

                # apply recursive ordering
                agent_mu, agent_log_var, lane_mu, lane_log_var, edge_index_lane_to_lane, _, _ = reorder_indices(
                    agent_mu,
                    agent_log_var,
                    lane_mu,
                    lane_log_var,
                    edge_index_lane_to_lane,
                    agent_states,
                    road_points,
                    scene_type,
                    dataset=self.cfg.dataset_name)
                edge_index_lane_to_lane = torch.from_numpy(edge_index_lane_to_lane)

                d['map_id'] = map_id
                d['lg_type'] = scene_type
                d['num_lanes'] = num_lanes
                d['num_agents'] = num_agents

                _, lane_latents = normalize_latents(
                    torch.empty((num_agents, self.cfg_model.agent_latent_dim)),
                    from_numpy(lane_mu),
                    self.cfg_dataset.agent_latents_mean,
                    self.cfg_dataset.agent_latents_std,
                    self.cfg_dataset.lane_latents_mean,
                    self.cfg_dataset.lane_latents_std
                )

                # these two are placeholders
                d['lane'].x = torch.empty((num_lanes, self.cfg_model.lane_latent_dim))
                d['agent'].x = torch.empty((num_agents, self.cfg_model.agent_latent_dim))

                # the lane latents will be used in the land-conditioned generation
                d['lane'].latents = lane_latents
                d['lane', 'to', 'lane'].edge_index = from_numpy(edge_index_lane_to_lane)
                d['agent', 'to', 'agent'].edge_index = from_numpy(edge_index_agent_to_agent)
                d['lane', 'to', 'agent'].edge_index = from_numpy(edge_index_lane_to_agent)
                data_list.append(d)

        # in inpainting mode, we still need to feed through the autoencoder to get latents
        # and construct the LDM pyg dataset object
        if mode == 'inpainting':
            data_list = self._build_ldm_dset_from_ae_dset_for_inpainting(data_list, batch_size, num_samples)

        conditioning_filenames = ([os.path.splitext(os.path.basename(f))[0] for f in conditioning_files]
                                  if conditioning_files is not None
                                  else None)
        return data_list, conditioning_filenames

    def setup(self, stage):
        self.train_dataset = WaymoDatasetAutoEncoder(self.cfg_dataset, split_name='train')
       # self.val_dataset = WaymoDatasetAutoEncoder(self.cfg_dataset, split_name='val')

        self.val_dataset, conditioning_filenames = self._initialize_pyg_dset(
            mode="initial_scene",
            num_samples=50000,
            batch_size=1024,
            conditioning_path=None,
            nocturne_compatible_only=False
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.train_batch_size, 
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=True,#worker_init_fn=worker_init_fn
                          persistent_workers=True
                          )


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          drop_last=False,
                          persistent_workers=True
                          )