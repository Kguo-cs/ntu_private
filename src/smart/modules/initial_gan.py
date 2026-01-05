import torch
import torch.nn as nn

from smart.utils import transform_to_global
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.distributions import Categorical

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer,CacheAttention
from src.smart.modules.edge_encoder import EdgeEncoder,topo_rank_among_edges
from torch_scatter import scatter_max,scatter_mean,scatter_sum
from src.smart.layers.relative_transformer import RoFormerBlock
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_local,
    transform_to_local,
    wrap_angle,
)

from .initial_discriminator import InitDiscriminator,InitGeneator


class InitGAN(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_freq_bands,
            token_processor,
    ) -> None:

        super(InitGAN, self).__init__()
        self.token_processor=token_processor

        self.hidden_dim=hidden_dim

        self.token_processor=token_processor

        self.D=InitDiscriminator(hidden_dim,num_heads,num_freq_bands,token_processor)

        self.G=InitGeneator(hidden_dim,num_heads,num_freq_bands,token_processor)

        self.global_step=0


    def forward(self,map_feature, tokenized_agent):

        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]
        batch=tokenized_agent["batch"]

        batch_num=tokenized_agent["num_graphs"]

        pred_mask = tokenized_agent["initial_ego_mask"]

        ego_position=tokenized_agent["initial_pos"][pred_mask]
        ego_heading=tokenized_agent["initial_heading"][pred_mask]

        pos_pl,orient_pl=transform_to_local(pos_pl[:,None],
                           orient_pl[:,None],
                           ego_position[batch_pl],
                           ego_heading[batch_pl],
                           )

        pos_pl=pos_pl[:,0]
        orient_pl=orient_pl[:,0]

        ego_dist=torch.linalg.norm(pos_pl,dim=-1)

        ego_dist_mask=ego_dist<100

        pos_pl=pos_pl[ego_dist_mask]
        orient_pl=orient_pl[ego_dist_mask]
        batch_pl=batch_pl[ego_dist_mask]
        feat_map=feat_map[ego_dist_mask]

        pos_pl, orient_pl, feat_map = self.padding(pos_pl, orient_pl, feat_map, batch_pl,batch_num)

        map_mask = torch.any(feat_map != 0, dim=-1)

        map_features=(pos_pl, orient_pl, feat_map, map_mask)

        criterion = nn.BCELoss()

        with torch.no_grad():
            fake_pos,fake_heading,fake_shape = self.G(map_features, tokenized_agent)

        real_labels = torch.ones(len(fake_pos), 1, device=fake_pos.device)
        fake_labels = torch.zeros(len(fake_pos), 1, device=fake_pos.device)

        real_pos,real_heading=transform_to_local(tokenized_agent["gt_initial_pos"],
                           tokenized_agent["gt_initial_pos"],
                           ego_position,
                           ego_heading,
                           )

        real_shape=tokenized_agent["shape"]
        real_pos=real_pos[:,0]
        real_heading=real_heading[:,0]

        real_loss = criterion(self.D(map_features,real_pos,real_heading,real_shape,tokenized_agent), real_labels)
        fake_loss = criterion(self.D(map_features,fake_pos,fake_heading,fake_shape,tokenized_agent), fake_labels)
        d_loss = real_loss + fake_loss

        g_loss = criterion(self.D(map_features,fake_pos,fake_heading,fake_shape,tokenized_agent), real_labels)

        return entry_logit
        # after interact with agent and map,  predict state and type and shape and tokenized position,
        # then refine predict head token and offset_xy ,
        # #then predict all agent motion
