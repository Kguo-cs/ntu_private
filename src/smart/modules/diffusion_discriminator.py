import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class MLPConditionDiffusion(nn.Module):
    def __init__(self, n_steps, cond_dim=6, data_dim=1, num_units=128, depth=4):
        super(MLPConditionDiffusion, self).__init__()
        self.data_dim = data_dim
        linears_list = []
        linears_list.append(nn.Linear(cond_dim + data_dim, num_units))
        linears_list.append(nn.ReLU())
        if depth > 1:
            for i in range(depth - 1):
                linears_list.append(nn.Linear(num_units, num_units))
                linears_list.append(nn.ReLU())
        linears_list.append(nn.Linear(num_units, data_dim))
        self.linears = nn.ModuleList(linears_list)#.to(device)

        embed_list = []
        for i in range(depth - 1):
            embed_list.append(nn.Embedding(n_steps, num_units))
        if depth == 1:
            embed_list.append(nn.Embedding(n_steps, num_units))
        self.step_embeddings = nn.ModuleList(embed_list)#.to(device)

    def forward(self, x, c, t):
        # print(x.shape, c.shape)
        x = torch.concat([x, c], dim=1)
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim,  base_net, num_units=128):
        super(Discriminator, self).__init__()
        input_dim = state_dim + action_dim
        #self.args = args
        self.base_net = False

        self.n_steps = n_steps = 1000
        betas = cosine_beta_schedule(self.n_steps)
        self.betas = betas#.to(self.args.device)
        alphas = 1 - betas
        alphas_prod = torch.cumprod(alphas, 0)
        alphas_bar_sqrt = torch.sqrt(alphas_prod)#.to(self.args.device)
        one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)#.to(self.args.device)

        d_model = MLPConditionDiffusion(n_steps, 1, input_dim, num_units=num_units,depth=1) #.to(self.args.device)
        try:
            self.base_net = base_net.net #.to(self.args.device)
        except:
            self.base_net = False
        self.model = d_model

        self.register_buffer("alphas_bar_sqrt",alphas_bar_sqrt)
        self.register_buffer("one_minus_alphas_bar_sqrt",one_minus_alphas_bar_sqrt)




    def diffusion_loss(self, label, sa_pair, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        batch_size = sa_pair.shape[0]

        # if self.sample_strategy == "constant":
        #     step = self.args.sample_strategy_value
        #     if step >= n_steps:
        #         step = n_steps - 1
        #     t = torch.full((batch_size,), step, device=sa_pair.device)
        #     t = t.unsqueeze(-1)
        # else:
        t = torch.randint(0, n_steps, size=(batch_size // 2,)).to(sa_pair.device)
        t = torch.cat([t, n_steps - 1 - t], dim=0)  # [batch_size, 1]
        t = t.unsqueeze(-1)

        # coefficient of x0
        a = alphas_bar_sqrt[t]

        # coefficient of eps
        aml = one_minus_alphas_bar_sqrt[t]
        label_input = torch.full((batch_size, 1), label).to(sa_pair.device)

        # generate random noise eps
        e = torch.randn_like(sa_pair).to(sa_pair.device)

        # model input
        x = sa_pair * a + e * aml

        # get predicted randome noise at time t
        output = self.model(x, label_input, t.squeeze(-1))

        return (e - output).square().mean(dim=1, keepdim=True)
        # return torch.unsqueeze(torch.mean(e - output, dim=1), 1)

    def diffusion_loss_fn(self, label, sa_pair):
        diff_loss = self.diffusion_loss(label, sa_pair, self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt,
                                        self.n_steps)

        return diff_loss

    def forward(self, state, action, label):

        if self.base_net:
            state = self.base_net(state)
        state_action = torch.cat([state, action], dim=1)
        loss = self.diffusion_loss_fn(label, state_action)
        return loss

    def _compute_disc_val(self, state, action, label=None):
        label_one = self.forward(state, action, 1.)
        label_zero = self.forward(state, action, 0.)
        output = F.softmax(torch.stack([-label_one, -label_zero]),dim=0)[0]
        return output

# ds=Discriminator(12, 2, False, num_units=128)
#
# state=torch.zeros([4,12])
# action=torch.zeros([4,2])
#
#
# ds=ds._compute_disc_val(state,action)
#
