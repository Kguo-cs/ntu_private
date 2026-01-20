import torch
import torch.nn as nn

from smart.utils import transform_to_global
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import padding
from smart.utils.earth_match import get_matching_loss
from src.smart.utils import (
    transform_to_local,
)

from src.smart.layers.initial_discriminator import InitDiscriminator,InitGeneator

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

        self.use_Rp=True

        self.Gamma =   1

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

        gt_initial_pos=tokenized_agent["initial_pos"]
        gt_initial_heading=tokenized_agent["initial_heading"]

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

        ego_traj=tokenized_agent["ego_traj"].reshape(len(ego_position),-1,2)

        ego_local_traj=transform_to_local(ego_traj,None,ego_position,ego_heading)[0]

        ego_embedding=self.G.ego_embedding(ego_local_traj.flatten(1,2))

        feat_map = feat_map + ego_embedding[batch_pl]

        pos_pl, orient_pl, feat_map = self.padding(pos_pl, orient_pl, feat_map, batch_pl,batch_num)

        map_mask = torch.any(feat_map != 0, dim=-1)

        padding_map_features=(pos_pl, orient_pl, feat_map, map_mask)

        non_ego = ~ego_mask

        batch = tokenized_agent["batch"][non_ego]

        if self.D.use_entry_former:
            map_feature = padding_map_features

        FakeSamples = self.G(padding_map_features, tokenized_agent)
        fake_pos = FakeSamples[:, :2]
        fake_heading = FakeSamples[:, 2]
        fake_shape = FakeSamples[:, 3:]

        agent_n=len(FakeSamples)

        if self.training:
            shape = tokenized_agent["initial_shape"]

            initial_vel=tokenized_agent["initial_vel"][non_ego]

            real_shape = shape[non_ego][:,:2]

            real_shape=torch.cat([real_shape,initial_vel],dim=-1)

            real_pos, real_heading = transform_to_local(gt_initial_pos[non_ego],
                                                        gt_initial_heading[non_ego],
                                                        ego_position[batch],
                                                        ego_heading[batch],
                                                        )

            RealSamples=torch.cat([real_pos, real_heading[:,None], real_shape],dim=-1)

            if self.global_step % 10 == 0:
                RealSamples = RealSamples.detach().requires_grad_(True)
                FakeSamples = FakeSamples.detach().requires_grad_(True)

                RealLogits = self.D(RealSamples, map_feature,tokenized_agent)
                FakeLogits = self.D(FakeSamples, map_feature,tokenized_agent)

                if self.Gamma>0:
                    R1Penalty = (self.Gamma / 2) * self.ZeroCenteredGradientPenalty(RealSamples, RealLogits)
                    R2Penalty =  (self.Gamma / 2) *self.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
                else:
                    R2Penalty = R1Penalty = torch.tensor(0.0, device=real_heading.device)


                if self.use_Rp:
                    RelativisticLogits = RealLogits - FakeLogits
                    AdversarialLoss = nn.functional.softplus(-RelativisticLogits).mean()
                else:
                    FakeLogits, fake_interact_logits=FakeLogits[:agent_n], FakeLogits[agent_n:]
                    RealLogits, real_interact_logits=RealLogits[:agent_n], RealLogits[agent_n:]

                    AdversarialLoss=FakeLogits.mean()-RealLogits.mean()
                    if len(fake_interact_logits)>0:
                        AdversarialLoss=AdversarialLoss+fake_interact_logits.mean()-real_interact_logits.mean()

                w=0.1#0.1+(1-self.global_step/10000.0)

                # R2Penalty=R1Penalty=torch.tensor(0.0, device=real_heading.device)

                loss = (AdversarialLoss ,w*R2Penalty.mean(),w*R1Penalty.mean())#cosine schedule
            else:
                self.D.eval()
                FakeLogits = self.D(FakeSamples, map_feature,tokenized_agent)

                if self.use_Rp:
                    RealLogits = self.D(RealSamples, map_feature,tokenized_agent)
                    RelativisticLogits = FakeLogits - RealLogits
                    AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
                    loss = AdversarialLoss.mean()

                else:
                    FakeLogits, fake_interact_logits=FakeLogits[:agent_n], FakeLogits[agent_n:]
                    loss=-FakeLogits.mean()
                    if len(fake_interact_logits)>0:
                        loss=loss-fake_interact_logits.mean()

                # loss = -self.D(FakeSamples,map_feature,tokenized_agent).mean()
                self.D.train()
                #loss=torch.tensor(0.0, device=real_heading.device)
                initial_type = tokenized_agent["initial_type"][non_ego]

                match_loss,pos_loss,heading_loss,shape_loss=get_matching_loss(initial_type, batch, FakeSamples,RealSamples,1,0)

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

            shape=tokenized_agent["shape"]

            shape[non_ego,:2]=fake_shape[:,:2]

            tokenized_agent["shape"]= shape

            initial_vel = tokenized_agent["initial_vel"]

            initial_vel[non_ego] = fake_shape[:, -2:]

            center_token_traj = tokenized_agent["token_traj"].mean(-2)

            gt_initial_idx = torch.linalg.norm(center_token_traj - initial_vel[:, None] * 0.5, dim=-1).argmin(-1)

            gt_initial_speed=initial_vel.norm(dim=-1)

            return gt_initial_pos[:, None], gt_initial_heading[:, None],gt_initial_idx[:, None],gt_initial_speed

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