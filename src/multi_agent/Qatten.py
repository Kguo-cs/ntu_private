import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QattenMixer(nn.Module):
    def __init__(self, hidden_dim,num_heads=4):
        super(QattenMixer, self).__init__()

        # self.args = args
        # self.n_agents = args.n_agents
        # self.state_dim = int(np.prod(args.state_shape))
        # self.u_dim = int(np.prod(args.agent_own_state_size))

        self.state_dim = hidden_dim #global state dim

        self.u_dim = hidden_dim #local state dim

        self.n_query_embedding_layer1 = 64
        self.n_query_embedding_layer2 = 32
        self.n_key_embedding_layer1 = 32
        self.n_head_embedding_layer1 = 64
        self.n_head_embedding_layer2 = num_heads
        self.n_attention_head = num_heads
        self.n_constrant_value = 32

        self.query_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(nn.Sequential(nn.Linear(self.state_dim, self.n_query_embedding_layer1),
                                                             nn.ReLU(),
                                                             nn.Linear(self.n_query_embedding_layer1,
                                                                       self.n_query_embedding_layer2)))

        self.key_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.key_embedding_layers.append(nn.Linear(self.u_dim, self.n_key_embedding_layer1))

        self.scaled_product_value = np.sqrt(self.n_query_embedding_layer2)

        self.type="weigthed"

        if self.type=="weigthed":

            self.head_embedding_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_head_embedding_layer1),
                                                  nn.ReLU(),
                                                  nn.Linear(self.n_head_embedding_layer1, self.n_head_embedding_layer2))

        self.constrant_value_layer = nn.Sequential(nn.Linear(self.state_dim, self.n_constrant_value),
                                                   nn.ReLU(),
                                                   nn.Linear(self.n_constrant_value, 1))


    def forward(self, agent_qs, states,agent_states,attention_mask):
        bs = agent_qs.size(0)
        n_agents=agent_qs.size(1)
        states = states.reshape(-1, self.state_dim) #batch_size,state_dim
        us = agent_states.reshape(-1, self.state_dim) #self._get_us(states)# batch_size*agent_num, state_dim
        agent_qs = agent_qs.view(-1, 1, n_agents)

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](states)
            u_embedding = self.key_embedding_layers[i](us)

            # shape: [-1, 1, state_dim]
            state_embedding = state_embedding.reshape(-1, 1, self.n_query_embedding_layer2)
            # shape: [-1, state_dim, n_agent]
            u_embedding = u_embedding.reshape(-1, n_agents, self.n_key_embedding_layer1)
            u_embedding = u_embedding.permute(0, 2, 1)

            # shape: [-1, 1, n_agent]
            attn = torch.matmul(state_embedding, u_embedding) / self.scaled_product_value

            #attn=torch.ones_like(attn)

            if attention_mask is not None:
                attn_bias = torch.where(attention_mask[:,None], -1e9, 0.)
                attn.add_(attn_bias)

            attn = F.softmax(attn, dim=-1)

            if attention_mask is not None:
                attn = attn.masked_fill(attention_mask[:,None], 0)

            q_lambda_list.append(attn)

        # shape: [-1, n_attention_head, n_agent]
        q_lambda_list = torch.stack(q_lambda_list, dim=1).squeeze(-2)

        # shape: [-1, n_agent, n_attention_head]
        q_lambda_list = q_lambda_list.permute(0, 2, 1)

        # shape: [-1, 1, n_attention_head]
        q_h = torch.matmul(agent_qs, q_lambda_list)

        if self.type == 'weigthed':
            # shape: [-1, n_attention_head, 1]
            w_h = torch.abs(self.head_embedding_layer(states))
            w_h = w_h.reshape(-1, self.n_head_embedding_layer2, 1)

            # shape: [-1, 1]
            sum_q_h = torch.matmul(q_h, w_h)
            sum_q_h = sum_q_h.reshape(-1, 1)
        else:
            # shape: [-1, 1]
            sum_q_h = q_h.sum(-1)
            sum_q_h = sum_q_h.reshape(-1, 1)

        c = 0#self.constrant_value_layer(states)
        q_tot = sum_q_h + c
        q_tot = q_tot.view(bs, -1, 1)
        return q_tot

    def _get_us(self, states):
        agent_own_state_size = self.args.agent_own_state_size
        with torch.no_grad():
            us = states[:, :agent_own_state_size * self.n_agents].reshape(-1, agent_own_state_size)
        return us