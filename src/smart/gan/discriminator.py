import torch
import torch.nn as nn

from smart.utils import transform_to_global
from src.smart.layers import MLPLayer
from src.smart.layers.relative_transformer import RoFormerBlock, padding
from torch.distributions import Categorical

from src.smart.layers import MLPLayer
from src.smart.layers.attention_layer import AttentionLayer, CacheAttention
from src.smart.modules.edge_encoder import EdgeEncoder
from torch_scatter import scatter_max, scatter_mean, scatter_sum
from src.smart.layers.relative_transformer import RoFormerBlock,RoFormerDecoder
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_local,
    transform_to_local,
    wrap_angle,
)
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
import copy
from torch import Tensor
from src.smart.loss.earth_match import get_matching_loss
from src.smart.loss.rollout_buffer import RunningMeanStdTorch


class InitDiscriminator(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_freq_bands,
            token_processor,
    ) -> None:

        super(InitDiscriminator, self).__init__()
        self.token_processor = token_processor

        self.hidden_dim = hidden_dim

        self.use_entry_former = False
        self.use_transformer=False
        self.use_decompose = True

        self.dis_weight=10
        self.dist_decay=3

        if self.use_entry_former:

            if self.use_transformer:
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=self.hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=self.hidden_dim * 4,
                    dropout=0,
                    norm_first=True,
                    batch_first=True  # nn.Transformer uses (seq_len, batch, dim)
                )

                self.transformer_decoder = nn.TransformerDecoder(
                    decoder_layer,
                    num_layers=1
                )
                self.pos_embedding = MLPLayer(2, hidden_dim, hidden_dim)
                self.head_embedding = MLPLayer(1, hidden_dim, hidden_dim)

            else:
                self.entry_his_len = 1000000

                self.entry_former = RoFormerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0,
                                                  hist_len=self.entry_his_len)  # replace with gnn
        else:

            self.edge_encoder = EdgeEncoder(hidden_dim,
                                            num_freq_bands,
                                            use_a2a=True,
                                            use_pl2a=True,
                                            discriminator=True
                                            )

            num_layers = 1

            self.pt2a_attn_layers = nn.ModuleList(
                [
                    AttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        head_dim=hidden_dim // num_heads,
                        dropout=0,
                        bipartite=True,
                        has_pos_emb=True,
                        #  gated_attention=discriminator,
                    )
                    for _ in range(num_layers)
                ]
            )
            if self.use_decompose:
                self.a2a_head= MLPLayer(
                    input_dim=hidden_dim*3, hidden_dim=hidden_dim, output_dim=1
                )
            else:

                self.a2a_attn_layers = nn.ModuleList(
                    [
                        AttentionLayer(
                            hidden_dim=hidden_dim,
                            num_heads=num_heads,
                            head_dim=hidden_dim // num_heads,
                            dropout=0,
                            bipartite=True,
                            has_pos_emb=True,
                            #  gated_attention=discriminator,
                        )
                        for _ in range(num_layers)
                    ]
                )

        self.type_embedding = nn.Embedding(3, hidden_dim)
        self.shape_embedding = MLPLayer(4, hidden_dim, hidden_dim)

        self.score_decoder = MLPLayer(hidden_dim, hidden_dim, 1)

        self.use_GAIL=True

        self.use_Rp=False

        if self.use_GAIL:
            self.return_meanstd = RunningMeanStdTorch(shape=(1))

        self.Gamma=1

        self.use_sde=False

        self.noise_level=0.7


    def ZeroCenteredGradientPenalty(self,Samples, Critics):
        Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True)
        return Gradient.square().sum([-1])

    def get_d_loss(self,FakeSamples,target,map_feature, tokenized_agent,gamma=0):
        agent_n = len(FakeSamples)

        if gamma>0:

            FakeSamples = FakeSamples.detach().requires_grad_(True)

        FakeLogits, fake_weight, end_index = self.forward(FakeSamples, map_feature, tokenized_agent)

        if gamma>0:
            Penalty = (gamma / 2) * self.ZeroCenteredGradientPenalty(FakeSamples, FakeLogits).mean()
        else:
            Penalty=torch.tensor(0.0,device=FakeSamples.device)

        if self.use_Rp:
            RelativisticLogits = RealLogits - FakeLogits
            dis_loss = nn.functional.softplus(-RelativisticLogits).mean()
        else:
            FakeLogits1, fake_interact_logits = FakeLogits[:agent_n], FakeLogits[agent_n:]

            if gamma>0:
                dis_loss = F.binary_cross_entropy_with_logits(FakeLogits1, torch.zeros_like(FakeLogits1)+target,
                                                       reduction='mean')
            else:
                dis_loss =(1-2*target)*   FakeLogits1.mean() #torch.mean(F.relu(1.0 +(1-2*target)* FakeLogits1))#0->1

            gen_rewards = FakeLogits1[:, 0]  ##torch.nn.functional.logsigmoid(FakeLogits1.mean(-1))

            if self.use_decompose:
                if gamma > 0:
                    fake_interact_bce_loss = F.binary_cross_entropy_with_logits(
                        fake_interact_logits,
                        torch.zeros_like(fake_interact_logits)+target,
                       # weight=fake_weight.detach(),
                        reduction='none'
                    )
                    if gamma>0:
                        fake_weight=fake_weight.detach()
                    fake_interact_bce_loss= (fake_interact_bce_loss * fake_weight).mean()
                else:
                    weight_logit = fake_interact_logits[:, 0]* fake_weight[:,0]

                    valid_interact_reward = (1-2*target)* scatter_sum(weight_logit, end_index, dim=0, dim_size=agent_n)#[:, 0]

                    fake_interact_bce_loss=valid_interact_reward.mean()

                    #fake_loss = torch.mean(F.relu(1.0 + (1 - 2 * target) * fake_interact_logits))  # 0->1

                    gen_rewards = gen_rewards.detach() + valid_interact_reward.detach()

                dis_loss = dis_loss + fake_interact_bce_loss

        return dis_loss, gen_rewards, Penalty, FakeLogits1

    def update_dis(self,logger,opt_D,inputs,FakeSamples):

        RealSamples, _,_, map_feature, tokenized_agent= inputs

        dis_loss, gen_rewards, r1, FakeLogits=self.get_d_loss(FakeSamples,0,map_feature, tokenized_agent,self.Gamma)
        expert_dis_loss, expert_rewards, r2, RealLogits=self.get_d_loss(RealSamples,1,map_feature, tokenized_agent,self.Gamma)

        loss = expert_dis_loss+dis_loss + r1 + r2

        logger("train/dis_los", dis_loss.item(), on_step=True, batch_size=1)
        logger("train/r1", r1.item(), on_step=True, batch_size=1)
        logger("train/r2", r2.item(), on_step=True, batch_size=1)
        logger("train/d_loss", loss.item(), on_step=True, batch_size=1)
        disc_val = torch.sigmoid(FakeLogits)

        logger("train/agent_disc_val", disc_val.mean().item(), on_step=True, batch_size=1)
        logger("train/agent_disc_val_std", disc_val.std().item(), on_step=True, batch_size=1)
        disc_val = torch.sigmoid(RealLogits)

        logger("train/expert_disc_val", disc_val.mean().item(), on_step=True, batch_size=1)
        logger("train/expert_disc_val_std", disc_val.std().item(), on_step=True, batch_size=1)

        opt_D.zero_grad()
        loss.backward()#retain_graph=True
        torch.nn.utils.clip_grad_norm_( self.parameters(),   max_norm=1   )
        opt_D.step()

        return gen_rewards,expert_rewards

    def gan_update(self,logger,optimizer,G,inputs):
        RealSamples,fake_samples, match_loss, map_feature, tokenized_agent= inputs

        opt_G, opt_D = optimizer

        gen_rewards,expert_rewards=self.update_dis(logger,opt_D,inputs,fake_samples.detach())

        g_loss, gen_rewards, r1, FakeLogits=self.get_d_loss(fake_samples,0,map_feature, tokenized_agent)


        loss=match_loss*0.1-g_loss

        logger("train/g_loss", g_loss.item(), on_step=True, batch_size=1)


        opt_G.zero_grad()
        loss.backward()#retain_graph=True
        torch.nn.utils.clip_grad_norm_( G.parameters(),   max_norm=1   )
        opt_G.step()


        return loss

    def gail_update(self,logger,optimizer,G,inputs):
        RealSamples, match_loss, map_feature, tokenized_agent= inputs

        with torch.no_grad():
            rollout_samples, x_list, z_list, t_list = G.sample(tokenized_agent, map_feature,infer_steps=10, eval_mask=None,noise_level=self.noise_level)

        opt_G, opt_D = optimizer

        gen_rewards,expert_rewards=self.update_dis(logger,opt_D,inputs,rollout_samples)

        self.return_meanstd.update(gen_rewards)

        advantages = self.return_meanstd.normalize(gen_rewards)

        loss = update_policy(logger, opt_G, G, inputs, z_list, t_list,advantages,expert_rewards)

        #rollout_n =3
        #num_mc_samples=8
        return loss

    def get_reward(self,samples,t,tokenized_agent, map_feature,key):

        #samples=torch.cat([samples,t],dim=-1)

        Logits, weight,end_index = self.forward(samples, map_feature, tokenized_agent, return_weight=True)

        agent_n = len(samples)
        ego_logits = Logits[:agent_n]
        interact_logits = Logits[agent_n:]

        if key == "expert":
            target=1
            reward = None

        else:
            target=0

            reward = ego_logits.detach()

            if self.use_decompose:

                weight_logit= interact_logits * weight

                valid_interact_reward=scatter_sum(weight_logit, end_index, dim=0,  dim_size=len(samples))

                reward = reward + valid_interact_reward.detach()


        bce_loss = F.binary_cross_entropy_with_logits(ego_logits, torch.zeros_like(ego_logits)+target,
                                                           reduction='mean')

        if self.use_decompose:
            interact_bce_loss = F.binary_cross_entropy_with_logits(interact_logits,
                                                                   torch.zeros_like(interact_logits) + target,
                                                                   weight=weight, reduction='mean')

            bce_loss=bce_loss+interact_bce_loss

        return bce_loss,reward

    def padding(self, pos, heading, feature, batch, num_graphs):
        lengths = torch.bincount(batch, minlength=num_graphs).tolist()

        padding_pos_a = padding(pos, lengths, padding_value=0)  # b, n, d
        padding_heading_a = padding(heading, lengths, padding_value=0)  # b, n, d
        padding_features_a = padding(feature, lengths, padding_value=0)  # b, n, d

        return padding_pos_a, padding_heading_a, padding_features_a

    def embed_input(self, initial_pos, initial_heading, initial_type, initial_shape, batch, num_graphs):
        type_embedding = self.type_embedding(initial_type)
       # pos_embedding = self.pos_embedding(initial_pos)
        #heading_embedding = self.head_embedding(initial_heading[:, None])
        shape_embedding = self.shape_embedding(initial_shape)

        feat_a = type_embedding + shape_embedding#+ heading_embedding + pos_embedding

        pos_a_b, heading_a_b, feat_a_b = self.padding(initial_pos, initial_heading, feat_a, batch, num_graphs)

        mask_a_b = torch.any(feat_a_b != 0, dim=-1)

        return pos_a_b, heading_a_b, feat_a_b, mask_a_b

    def forward(self,inputs, map_feature,  tokenized_agent):

        #inputs=inputs+torch.randn_like(inputs)*1e-2
        pos_a=inputs[...,:2]
        head_a=torch.atan2(inputs[...,3],inputs[...,2])
        shape=inputs[...,4:]

        batch = tokenized_agent["nonego_batch"]
        type = tokenized_agent["nonego_type"]
        num_graphs = tokenized_agent["num_graphs"]
       # ego_embedding = tokenized_agent["ego_embedding"].detach()

        if self.use_entry_former:
            head_a = wrap_angle(head_a)

            pos_pl, orient_pl, feat_map, map_mask = map_feature

            pos_a_b, heading_a_b, feat_a_b, mask_a_b = self.embed_input(pos_a, head_a, type, shape, batch, num_graphs)

            feat_map=feat_map.detach()

           # feat_map = feat_map + self.pos_embedding(pos_pl) + self.head_embedding(orient_pl[:, :, None])

            if self.use_transformer:
                with torch.backends.cuda.sdp_kernel(
                        enable_mem_efficient=False,
                ):

                    attr_feature = self.transformer_decoder(
                        tgt=feat_a_b,  # self-attention queries
                        memory=feat_map,  # cross-attention keys/values
                        tgt_key_padding_mask=~mask_a_b,
                        memory_key_padding_mask=~map_mask
                    )
            else:

                attr_feature =self.entry_former(feat_a_b, pos_a_b,  heading_a_b, mask_a_b,
                                                                  feat_map,
                                                                  pos_pl,
                                                                  orient_pl, map_mask)
                # entry_feature = self.entry_former.cross_attention(feat_a_b, pos_a_b,
                #                                                   heading_a_b, mask_a_b,
                #                                                   feat_map,
                #                                                   pos_pl,
                #                                                   orient_pl, map_mask)
                #
                # attr_feature = self.attr_former.temporal_embed(entry_feature, pos_a_b, heading_a_b, 0, 0, mask_a_b,
                #                                                use_time=False,use_causal=False)

            attr_feature = attr_feature[mask_a_b]
        else:
            batch_pl = map_feature["batch"]
            pos_pl = map_feature["position"]
            orient_pl = map_feature["orientation"]
            feat_map = map_feature["pt_token"]

            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

            edge_index_a2a, r_a2a, dist, relative_pos, r_a2a_nei, center_nei_pos, center_nei_heading = self.edge_encoder.build_interaction_edge(
                pos_s=pos_a,  # [n_agent, n_step, 2]
                head_s=head_a,  # [n_agent, n_step]
                head_vector_s=head_vector_a,  # [n_agent, n_step, 2]
                batch_s=batch,  # [n_agent*n_step]
                mask=None,  # [n_agent, n_step]
                max_radius=60,
                max_num_neighbors=20,
                agent_train_mask=None,
                layer_num=1,
                counter_feat_a=None
            )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

            if batch_pl.max().item() != num_graphs - 1:
                batch = tokenized_agent["repeat_batch"]

                n_step = batch.shape[1]

                pos_b = pos_a.reshape(n_step, -1, 2)
                theta_b = head_a.reshape(n_step, -1)

                pos_a = pos_b.transpose(0, 1)
                head_a = theta_b.transpose(0, 1)

                head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

                mask = torch.ones_like(batch).to(torch.bool)
            else:
                mask = None

            edge_index_pl2a, r_pl2a = self.edge_encoder.build_map2agent_edge(
                pos_pl=pos_pl,  # [n_pl, 2]
                orient_pl=orient_pl,  # [n_pl]
                pos_a=pos_a,  # [n_agent, n_step, 2]
                head_a=head_a,  # [n_agent, n_step]
                head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
                mask=mask,  # [n_agent, n_step]
                batch_s=batch,  # [n_agent,n_step]
                batch_pl=batch_pl,  # [n_pl*n_step]
                pl2a_radius=40,
                max_num_neighbors=20,
                agent_train_mask=None,
                layer_num=1
            )

            type_embedding = self.type_embedding(type)
            shape_embedding = self.shape_embedding(shape)

            feat_a = type_embedding + shape_embedding#+ego_embedding

            if self.use_decompose:
                start_index = edge_index_a2a[0]       #edge_index[1] = src indices = its k nearest neighbors
                end_index = edge_index_a2a[1]        #edge_index[0] = dst indices = query point

                start_edge_feature = feat_a[start_index]
                end_edge_feature = feat_a[end_index]

                feat_interact = torch.cat([start_edge_feature, r_a2a, end_edge_feature], dim=-1)
                interact_logits = self.a2a_head(feat_interact)
            else:
                feat_a = self.a2a_attn_layers[0](feat_a, r_a2a, edge_index_a2a)
                end_index = None

            attr_feature = self.pt2a_attn_layers[0]((feat_map, feat_a), r_pl2a,
                                                    edge_index_pl2a)  # edge_index_pl2a[0] is the src, edge_index_pl2a[1] is dst

        score = self.score_decoder(attr_feature)

        if self.use_decompose:
            score=torch.cat([score, interact_logits], dim=0)

            weight = torch.exp(-dist[:, None] / self.dist_decay) * self.dis_weight  # torch.ones_like(dist) #=
        else:
            weight = None

        return score, weight, end_index
