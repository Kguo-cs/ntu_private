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
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

def matching_loss(
    fake_pos, fake_heading, fake_shape,
    real_pos, real_heading, real_shape,
    w_pos=0.1, w_heading=0.5, w_shape=0.5
):
    # Position: L1 or L2

    dist=torch.linalg.norm(fake_pos-real_pos,dim=-1)

    pos_loss = dist.mean()

    # Heading: periodic-safe loss
    # heading_diff = torch.atan2(
    #     torch.sin(fake_heading - real_heading),
    #     torch.cos(fake_heading - real_heading)
    # )
    heading_diff=wrap_angle(fake_heading - real_heading)
    heading_loss = heading_diff.abs().mean()

    # Shape: L1
    shape_loss = F.l1_loss(fake_shape, real_shape)

    total_loss = (
        w_pos * pos_loss +
        w_heading * heading_loss +
        w_shape * shape_loss
    )

    return total_loss,pos_loss,heading_loss,shape_loss

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

        self.D=InitDiscriminator(hidden_dim,num_heads,num_freq_bands,token_processor)

        self.G=InitGeneator(hidden_dim,num_heads,num_freq_bands,token_processor)

        self.global_step=0

        self.criterion = nn.BCELoss()

        self.use_Rp=False

    def padding(self,pos,heading,feature,batch,batch_num):
        lengths = torch.bincount(batch,minlength=batch_num).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        return padding_pos_a, padding_heading_a, padding_features_a

    def forward(self,map_feature, tokenized_agent):

        batch_pl = map_feature["batch"]
        pos_pl = map_feature["position"]
        orient_pl = map_feature["orientation"]
        feat_map = map_feature["pt_token"]

        batch_num=tokenized_agent["num_graphs"]

        gt_initial_pos=tokenized_agent["gt_initial_pos"][:,0]
        gt_initial_heading=tokenized_agent["gt_initial_heading"][:,0]

        ego_mask=tokenized_agent["ego_mask"]

        ego_position=gt_initial_pos[ego_mask]
        ego_heading=gt_initial_heading[ego_mask]

        pos_pl,orient_pl=transform_to_local(pos_pl,#[:,None],
                           orient_pl,#[:,None],
                           ego_position[batch_pl],
                           ego_heading[batch_pl],
                           )

        ego_dist=torch.linalg.norm(pos_pl,dim=-1)

        ego_dist_mask=ego_dist<100 #120

        pos_pl=pos_pl[ego_dist_mask]
        orient_pl=orient_pl[ego_dist_mask]
        batch_pl=batch_pl[ego_dist_mask]
        feat_map=feat_map[ego_dist_mask]

        map_feature=(pos_pl,orient_pl,batch_pl,feat_map)

        pos_pl, orient_pl, feat_map = self.padding(pos_pl, orient_pl, feat_map, batch_pl,batch_num)

        map_mask = torch.any(feat_map != 0, dim=-1)

        padding_map_features=(pos_pl, orient_pl, feat_map, map_mask)

        non_ego = ~ego_mask

        batch = tokenized_agent["batch"][non_ego]

        shape=tokenized_agent["shape"]

        real_shape=shape[non_ego]

        real_pos, real_heading = transform_to_local(gt_initial_pos[non_ego],
                                                    gt_initial_heading[non_ego],
                                                    ego_position[batch],
                                                    ego_heading[batch],
                                                    )

        if self.D.use_entry_former:
            map_feature = padding_map_features

        RealSamples=torch.cat([real_pos, real_heading[:,None], real_shape],dim=-1)
        FakeSamples = self.G(padding_map_features, tokenized_agent)
        fake_pos = FakeSamples[:, :2]
        fake_heading = FakeSamples[:, 2]
        fake_shape = FakeSamples[:, 3:]

        if self.training:

            if self.global_step%10==0:
                RealSamples = RealSamples.detach().requires_grad_(True)
                FakeSamples = FakeSamples.detach().requires_grad_(True)

                RealLogits = self.D(RealSamples, map_feature,tokenized_agent)
                FakeLogits = self.D(FakeSamples, map_feature,tokenized_agent)

                Gamma = 1

                R1Penalty = (Gamma / 2) * self.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
                R2Penalty =  (Gamma / 2) *self.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)

                if self.use_Rp:
                    RelativisticLogits = RealLogits - FakeLogits
                    AdversarialLoss = nn.functional.softplus(-RelativisticLogits).mean()
                else:
                    AdversarialLoss=FakeLogits.mean()-RealLogits.mean()

                # Gamma=1
                #
                # DiscriminatorLoss = AdversarialLoss + (Gamma / 2) * (R1Penalty + R2Penalty)

                # with torch.no_grad():
                #     fake_pos, fake_heading, fake_shape = self.G(padding_map_features, tokenized_agent)
                #
                #
                # real_loss = -self.D(map_feature, real_pos, real_heading, real_shape, tokenized_agent).mean()
                # fake_loss = self.D(map_feature,fake_pos,fake_heading,fake_shape,tokenized_agent).mean()
                #
                #
                # inputs=torch.cat([real_pos, real_heading[:,None], real_shape],dim=-1)
                #
                # r1=self.compute_gp(map_feature,inputs,tokenized_agent)
                # inputs=torch.cat([fake_pos, fake_heading[:,None], fake_shape],dim=-1)
                # r2=self.compute_gp(map_feature,inputs,tokenized_agent)

                # gp=r1+r2#

                w=1#0.1+(1-self.global_step/10000.0)

                # R2Penalty=R1Penalty=torch.tensor(0.0, device=real_heading.device)

                loss = (AdversarialLoss ,w*R2Penalty.mean(),w*R1Penalty.mean())#cosine schedule
            else:
                self.D.eval()
                FakeLogits = self.D(FakeSamples, map_feature,tokenized_agent)

                if self.use_Rp:
                    RealLogits = self.D(RealSamples, map_feature,tokenized_agent)
                    RelativisticLogits = FakeLogits - RealLogits
                    AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
                else:
                    AdversarialLoss=-FakeLogits

                loss=AdversarialLoss.mean()

                # loss = -self.D(FakeSamples,map_feature,tokenized_agent).mean()
                self.D.train()
                #loss=torch.tensor(0.0, device=real_heading.device)

                rows, cols = [], []
                initial_type = tokenized_agent["type"][non_ego]

                for b in batch.unique():
                    for type in initial_type[batch == b].unique():
                        f_idx = ((batch == b) & (initial_type==type)).nonzero(as_tuple=True)[0]

                        dist = torch.cdist(fake_pos[f_idx], real_pos[f_idx])

                        cost = dist.cpu().detach().numpy()

                        row, col = linear_sum_assignment(cost)

                        rows.append(f_idx[row])
                        cols.append(f_idx[col])

                row = torch.cat(rows)
                col = torch.cat(cols)

                match_loss,pos_loss,heading_loss,shape_loss = matching_loss(
                    fake_pos[row], fake_heading[row], fake_shape[row],
                    real_pos[col], real_heading[col], real_shape[col]
                )

                #match_loss=(1-self.global_step/10000.0)*match_loss

                #match_loss=pos_loss=heading_loss=shape_loss=torch.tensor(0.0, device=real_heading.device)

                loss=(loss,match_loss,pos_loss,heading_loss,shape_loss)

            self.global_step+=1
            return loss
        else:
            global_pos,global_heading=transform_to_global(
                fake_pos,
                fake_heading,
                ego_position[batch],
                ego_heading[batch],
            )

            gt_initial_pos[non_ego]=global_pos
            gt_initial_heading[non_ego]=global_heading
            shape[non_ego]=fake_shape

            tokenized_agent["shape"]= shape
            # tokenized_agent["ego_mask"] = tokenized_agent["initial_ego_mask"]
            # tokenized_agent["type"] = tokenized_agent['initial_type']
            # tokenized_agent['id']=tokenized_agent['initial_id']

            return gt_initial_pos[:,None], gt_initial_heading[:,None]

    def ZeroCenteredGradientPenalty(self,Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([-1])


    def compute_gp(self,map_feature,inputs,tokenized_agent):
        inputs.requires_grad_(True)  # IMPORTANT

        logit = self.D(map_feature, inputs[:, :2], inputs[:, 2], inputs[:, 3:6], tokenized_agent)

        disc_flat = logit.reshape(-1, 1)
        grad_outputs = torch.ones_like(disc_flat)

        # Compute gradients wrt interpolated inputs
        grad_all = torch.autograd.grad(
            outputs=disc_flat,  # whatever you use
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_norm = grad_all.norm(2, dim=1)  # [B]
        gp_lambda = 1

        gp = (grad_norm ** 2).mean() * gp_lambda / 2
        return gp
        # after interact with agent and map,  predict state and type and shape and tokenized position,
        # then refine predict head token and offset_xy ,
        # #then predict all agent motion
