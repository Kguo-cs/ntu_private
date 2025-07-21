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


# class SMART_GAIL(GAIL, SMART):
#     def __init__(self, model_config) -> None:
#         SMART.__init__(self, model_config)  # Explicit call
#         GAIL.__init__(self, model_config)  # Explicit call
#
#         model_config.decoder.hidden_dim=model_config.decoder.hidden_dim//2
#
#         self.discriminator=SMARTDecoder(
#             **model_config.decoder, n_token_agent=self.token_processor.n_token_agent )

class SMART_IQ(IQ_SoftQ, SMART):
    def __init__(self, model_config) -> None:
        SMART.__init__(self, model_config)  # Explicit call
        IQ_SoftQ.__init__(self, model_config)  # Explicit call


    def configure_optimizers(self):
        if  self.automatic_optimization:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

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

            lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            return [optimizer], [lr_scheduler]

        else:
            actor_optimizer = torch.optim.Adam(list(self.encoder.map_encoder.parameters())+list(self.encoder.agent_encoder.parameters())
                                               +list(self.encoder.value_network.parameters())
                                               , lr=self.lr)
            critic_optimizer = torch.optim.Adam(self.encoder.discriminator.parameters(), lr=self.lr)

            return [actor_optimizer, critic_optimizer]





