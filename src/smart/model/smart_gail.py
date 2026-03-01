# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from src.smart.model.iq_learn import IQ_SoftQ
from src.smart.model.smart import SMART
from src.smart.modules.smart_decoder import SMARTDecoder

import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from src.smart.model.optimizer import configure_optimizers,CombinedOptimizer,Muon
import torch.optim as optim


class SMART_IQ(IQ_SoftQ, SMART):
    def __init__(self, model_config) -> None:
        SMART.__init__(self, model_config)  # Explicit call
        IQ_SoftQ.__init__(self, model_config)  # Explicit call


    def configure_optimizers(self):
        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                        self.lr_min_ratio
                        + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                    1.0
                    + math.cos(
                math.pi
                * min(
                    1.0,
                    (current_step - self.lr_warmup_steps)
                    / (self.lr_total_steps - self.lr_warmup_steps),
                )
            )
            )

        if  self.automatic_optimization:

            # if self.encoder.use_gail and self.encoder.iq_learn:
            #     # policy_optimizer = torch.optim.AdamW(list(self.encoder.map_encoder.parameters())+list(self.encoder.agent_encoder.parameters())  , lr=self.lr)
            #     # discriminator_optimizer = torch.optim.AdamW(self.encoder.discriminator.parameters(),weight_decay=1, lr=3e-4)
            #     # value_optimizer = torch.optim.AdamW(list(self.encoder.value_network.parameters())+list(self.encoder.nei_value_network.parameters()), lr=3e-4)
            #     #
            #     # lr_scheduler = LambdaLR(policy_optimizer, lr_lambda=lr_lambda)
            #     #
            #     # return (
            #     #     [policy_optimizer, discriminator_optimizer,value_optimizer],
            #     #     [lr_scheduler, None,None],  # no scheduler for discriminator
            #     # )
            #     optimizer = torch.optim.AdamW(
            #         [
            #             {"params": list(self.encoder.map_encoder.parameters())+list(self.encoder.agent_encoder.parameters()), "lr": self.lr, "weight_decay": 0.01},
            #             {"params": self.encoder.discriminator.parameters(), "lr": 3e-5, "weight_decay": 0.01},
            #             {"params": list(self.encoder.value_network.parameters())+list(self.encoder.nei_value_network.parameters()), "lr": 3e-4, "weight_decay": 0.01},
            #         ]
            #     )
            #
            # else:
     #        hidden_params = []
     #        other_params = []
     #
     #        for name, p in self.encoder.named_parameters():
     #            if not p.requires_grad:
     #                continue
     #
     #            if p.ndim == 2 and "embed" not in name and "head" not in name:
     #                hidden_params.append(p)
     #            else:
     #                other_params.append(p)
     #
     #        muon_lr = 0.02 * (self.lr / 1e-4)
     #
     #        muon_opt = Muon(
     #            hidden_params,
     #            lr=muon_lr,
     #            momentum=self.muon_momentum,
     #        )
     #
     #        adamw_opt = optim.AdamW(
     #            other_params,
     #            lr=self.lr,
     #            weight_decay=self.weight_decay,
     #            betas=(0.9, 0.999),
     #        )
     #
     #        optimizers_list=[muon_opt, adamw_opt]
     #
     #        schedulers = []
     #        for opt in optimizers_list:
     # #           sched = get_cosine_schedule_with_warmup(
     #  #              opt, num_warmup_steps=self.lr_warmup_steps, num_training_steps=total_steps
     #   #         )
     #            sched=LambdaLR(opt, lr_lambda=lr_lambda)
     #
     #            schedulers.append(sched)
     #
     #        # Lightning expects a list
     #        return optimizers_list,schedulers

            optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr)

            lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

            return [optimizer], [lr_scheduler]

        else:
            # actor_optimizer = torch.optim.AdamW(list(self.encoder.agent_encoder.parameters())  +list(self.encoder.value_network.parameters()) , lr=self.lr)
            # discriminator_optimizer = torch.optim.AdamW(self.encoder.discriminator.parameters(), lr=self.lr)
            # # actor_optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr)
            # # lcf_optimizer = torch.optim.Adam(self.lcf_parameters.parameters(), lr=self.lr)
            # #list(self.encoder.map_encoder.parameters())+

            actor_optimizer=torch.optim.Adam(list(self.encoder.map_encoder.parameters())+list(self.encoder.agent_encoder.interative_decoder.parameters())+
                                             list(self.encoder.agent_encoder.interative_decoder.parameters()) +  list(self.encoder.agent_encoder.agent_token_embedding.parameters())+
                                             list( self.encoder.agent_encoder.init_decoder.G.parameters()), lr=self.lr)#,betas=(0.0,0.0)
            discriminator_optimizer=torch.optim.Adam(self.encoder.agent_encoder.init_decoder.D.parameters(), lr=self.lr)#,weight_decay=10

            return [actor_optimizer, discriminator_optimizer]
            # optimizers_list = configure_optimizers(self.encoder, muon_lr=0.02 * (self.lr / 1e-4), adam_lr=self.lr)
            #
            # optimizer = CombinedOptimizer(optimizers_list)
            #
            # schedulers = []
            # for opt in optimizers_list:
            #     # sched = get_cosine_schedule_with_warmup(
            #     #     opt, num_warmup_steps=self.lr_warmup_steps, num_training_steps=total_steps
            #     # )
            #     sched= LambdaLR(opt, lr_lambda)
            #     schedulers.append(sched)
            #
            #
            # return [optimizer],[schedulers]
            #




