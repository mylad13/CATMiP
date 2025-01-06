import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
# from .distributions import Bernoulli, Categorical, DiagGaussian
from hetmarl.algorithms.utils.util import check, init
from hetmarl.algorithms.utils.transformer_act import discrete_autoregreesive_act
from hetmarl.algorithms.utils.transformer_act import discrete_parallel_act
from hetmarl.algorithms.utils.transformer_act import continuous_autoregreesive_act
from hetmarl.algorithms.utils.transformer_act import continuous_parallel_act
from hetmarl.algorithms.utils.transformer_act import multidiscrete_autoregreesive_act
from hetmarl.algorithms.utils.transformer_act import multidiscrete_parallel_act
from hetmarl.algorithms.utils.channel_vit import ChannelVisionTransformer
from functools import partial

import icecream as ic

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
        #                      .view(1, 1, n_agent + 1, n_agent + 1))
        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()
        # print("B, L, D is: ", B, L, D)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)
        
        n_agents = L
        
        
        if self.masked: #adaptive masked attention
            mask = torch.tril(torch.ones(n_agents + 1, n_agents + 1)).view(1, 1, n_agents + 1, n_agents + 1)
            if torch.cuda.is_available():
                mask = mask.to("cuda:0")
            # att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
            att = att.masked_fill(mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_embd_vit, n_head, n_agent, n_agent_types, encode_state):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state

        self.active_func1 = nn.ReLU()
        self.active_func2 = nn.GELU()
        
        global_input_channel = obs_dim['global_agent_map'].shape[0]
        global_input_width = obs_dim['global_agent_map'].shape[1]
        global_input_height = obs_dim['global_agent_map'].shape[2]
        fixed_width = 8
        fixed_height = 8

        global_out_channel1 = 32
        global_out_channel2 = 16
        global_out_channel3 = 16

        global_kernel_size1 = 7
        global_kernel_size2 = 3
        global_kernel_size3 = 3

        global_stride1 = 1
        global_stride2 = 1
        global_stride3 = 1

        global_padding_size1 = 3
        global_padding_size2 = 1
        global_padding_size3 = 1

        fixed_image_size = fixed_width*fixed_height
        global_flattened_size1 = global_out_channel1*fixed_image_size
        global_flattened_size2 = global_out_channel2*fixed_image_size
        global_flattened_size3 = global_out_channel3*fixed_image_size

        global_output_width1 = (global_input_width-global_kernel_size1+global_padding_size1*2)//global_stride1 + 1
        global_output_height1 = (global_input_height-global_kernel_size1+global_padding_size1*2)//global_stride1 + 1
        global_out_image_size1 = global_output_width1*global_output_height1
        global_output_width2 = (global_output_width1-global_kernel_size2+global_padding_size2*2)//global_stride2 + 1
        global_output_height2 = (global_output_height1-global_kernel_size2+global_padding_size2*2)//global_stride2 + 1
        global_out_image_size2 = global_output_width2*global_output_height2
        global_output_width3 = (global_output_width2-global_kernel_size3+global_padding_size3*2)//global_stride3 + 1
        global_output_height3 = (global_output_height2-global_kernel_size3+global_padding_size3*2)//global_stride3 + 1

        conv_out_image_size3 = global_output_width3*global_output_height3
        
        global_hidden_size1 = n_embd
        # global_hidden_size2 = 2000
        # global_hidden_size3 = 500



        local_input_channel = obs_dim['local_agent_map'].shape[0]
        local_input_width = obs_dim['local_agent_map'].shape[1]
        local_input_height = obs_dim['local_agent_map'].shape[2]

        local_out_channel1 = 64
        local_kernel_size1 = 3
        local_stride1 = 1
        local_padding_size1 = 1

        local_out_channel2 = 32
        local_kernel_size2 = 3
        local_stride2 = 1
        local_padding_size2 = 1

        local_out_channel3 = 16
        local_kernel_size3 = 3
        local_stride3 = 1
        local_padding_size3 = 1
        
        local_ouput_width1 = (local_input_width-local_kernel_size1+local_padding_size1*2)//local_stride1 + 1
        local_output_height1 = (local_input_height-local_kernel_size1+local_padding_size1*2)//local_stride1 + 1
        local_out_image_size1 = local_ouput_width1*local_output_height1
        local_ouput_width2 = (local_ouput_width1-local_kernel_size2+local_padding_size2*2)//local_stride2 + 1
        local_output_height2 = (local_output_height1-local_kernel_size2+local_padding_size2*2)//local_stride2 + 1
        local_out_image_size2 = local_ouput_width2*local_output_height2
        local_ouput_width3 = (local_ouput_width2-local_kernel_size3+local_padding_size3*2)//local_stride3 + 1
        local_output_height3 = (local_output_height2-local_kernel_size3+local_padding_size3*2)//local_stride3 + 1
        local_out_image_size3 = local_ouput_width3*local_output_height3

        local_flattened_size1 = local_out_channel1*local_out_image_size1
        local_flattened_size2 = local_out_channel2*local_out_image_size2
        local_flattened_size3 = local_out_channel3*local_out_image_size3

        local_hidden_size1 = n_embd
        hidden_size1 = n_embd        

        print("local_flattened_size2 is: ", local_flattened_size2)
        print("global_flattened_size1 is: ", global_flattened_size1)
    

        self.agent_class_encoder = nn.Sequential(init_(nn.Linear(n_agent_types, n_embd), activate=True), self.active_func2)        
        
        # # Map-Size-Adaptive Observations Encoder (1-1 Layers)
        # self.global_obs_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=global_input_channel, out_channels=global_out_channel1, kernel_size=global_kernel_size1, stride=global_stride1, padding=global_padding_size1)), 
        #     nn.AdaptiveMaxPool2d(fixed_width), nn.LayerNorm([global_out_channel1, fixed_width, fixed_height]), self.active_func1, nn.Flatten())
        
        
        # self.local_obs_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=local_input_channel, out_channels=local_out_channel1, kernel_size=local_kernel_size1, stride=local_stride1, padding=local_padding_size1)), 
        #     nn.LayerNorm([local_out_channel1, local_ouput_width1, local_output_height1]), self.active_func1, nn.Flatten())
        
        # self.obs_encoder = nn.Sequential(
        #     init_(nn.Linear(local_flattened_size1+global_flattened_size1, hidden_size1), activate=True), self.active_func2, nn.LayerNorm(hidden_size1),
        #     init_(nn.Linear(hidden_size1, n_embd), activate=True), self.active_func2)
        
        
        # Map-Size-Adaptive Observations Encoder (1-2 Layers)
        self.global_obs_encoder = nn.Sequential(
            init_(nn.Conv2d(in_channels=global_input_channel, out_channels=global_out_channel1, kernel_size=global_kernel_size1, stride=global_stride1, padding=global_padding_size1)), 
            nn.AdaptiveMaxPool2d(fixed_width), nn.LayerNorm([global_out_channel1, fixed_width, fixed_height]), self.active_func1, nn.Flatten())
        
        
        self.local_obs_encoder = nn.Sequential(
            init_(nn.Conv2d(in_channels=local_input_channel, out_channels=local_out_channel1, kernel_size=local_kernel_size1, stride=local_stride1, padding=local_padding_size1)), 
            nn.LayerNorm([local_out_channel1, local_ouput_width1, local_output_height1]), self.active_func1,
            init_(nn.Conv2d(in_channels=local_out_channel1, out_channels=local_out_channel2, kernel_size=local_kernel_size2, stride=local_stride2, padding=local_padding_size2)), 
            nn.LayerNorm([local_out_channel2, local_ouput_width2, local_output_height2]), self.active_func1, nn.Flatten())
        
        self.obs_encoder = nn.Sequential(
            init_(nn.Linear(local_flattened_size2+global_flattened_size1, hidden_size1), activate=True), self.active_func2, nn.LayerNorm(hidden_size1),
            init_(nn.Linear(hidden_size1, n_embd), activate=True), self.active_func2)
        
        # # Map-Size-Adaptive Observations Encoder (2-2 Layers)
        # self.global_obs_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=global_input_channel, out_channels=global_out_channel1, kernel_size=global_kernel_size1, stride=global_stride1, padding=global_padding_size1)), 
        #     nn.AdaptiveMaxPool2d(fixed_width), nn.LayerNorm([global_out_channel1, fixed_width, fixed_height]), self.active_func1,
        #     init_(nn.Conv2d(in_channels=global_out_channel1, out_channels=global_out_channel2, kernel_size=global_kernel_size2, stride=global_stride2, padding=global_padding_size2)),
        #     nn.LayerNorm([global_out_channel2, fixed_width, fixed_height]), self.active_func1, nn.Flatten())
        
        
        # self.local_obs_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=local_input_channel, out_channels=local_out_channel1, kernel_size=local_kernel_size1, stride=local_stride1, padding=local_padding_size1)), 
        #     nn.LayerNorm([local_out_channel1, local_ouput_width1, local_output_height1]), self.active_func1,
        #     init_(nn.Conv2d(in_channels=local_out_channel1, out_channels=local_out_channel2, kernel_size=local_kernel_size2, stride=local_stride2, padding=local_padding_size2)), 
        #     nn.LayerNorm([local_out_channel2, local_ouput_width2, local_output_height2]), self.active_func1, nn.Flatten())
        
        # self.obs_encoder = nn.Sequential(
        #     init_(nn.Linear(local_flattened_size2+global_flattened_size2, hidden_size1), activate=True), self.active_func2, nn.LayerNorm(hidden_size1),
        #     init_(nn.Linear(hidden_size1, n_embd), activate=True), self.active_func2)
        
        # # Map-Size-Adaptive Observations Encoder (3-3 Layers)
        # self.global_obs_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=global_input_channel, out_channels=global_out_channel1, kernel_size=global_kernel_size1, stride=global_stride1, padding=global_padding_size1)), 
        #     nn.LayerNorm([global_out_channel1, global_output_width1, global_output_height1]), self.active_func1,
        #     init_(nn.Conv2d(in_channels=global_out_channel1, out_channels=global_out_channel2, kernel_size=global_kernel_size2, stride=global_stride2, padding=global_padding_size2)),
        #     nn.LayerNorm([global_out_channel2, global_output_width2, global_output_height2]), self.active_func1,
        #     init_(nn.Conv2d(in_channels=global_out_channel2, out_channels=global_out_channel3, kernel_size=global_kernel_size3, stride=global_stride3, padding=global_padding_size3)),
        #     nn.AdaptiveMaxPool2d(fixed_width), nn.LayerNorm([global_out_channel3, fixed_width, fixed_height]), self.active_func1, nn.Flatten())
        
        
        # self.local_obs_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=local_input_channel, out_channels=local_out_channel1, kernel_size=local_kernel_size1, stride=local_stride1, padding=local_padding_size1)), 
        #     nn.LayerNorm([local_out_channel1, local_ouput_width1, local_output_height1]), self.active_func1,
        #     init_(nn.Conv2d(in_channels=local_out_channel1, out_channels=local_out_channel2, kernel_size=local_kernel_size2, stride=local_stride2, padding=local_padding_size2)), 
        #     nn.LayerNorm([local_out_channel2, local_ouput_width2, local_output_height2]), self.active_func1,
        #      init_(nn.Conv2d(in_channels=local_out_channel2, out_channels=local_out_channel3, kernel_size=local_kernel_size3, stride=local_stride3, padding=local_padding_size3)), 
        #     nn.LayerNorm([local_out_channel3, local_ouput_width3, local_output_height3]), self.active_func1, nn.Flatten())

        
        # self.obs_encoder = nn.Sequential(
        #     init_(nn.Linear(local_flattened_size3+global_flattened_size3, hidden_size1), activate=True), self.active_func2, nn.LayerNorm(hidden_size1),
        #     init_(nn.Linear(hidden_size1, n_embd), activate=True), self.active_func2)
        
        


        # state encoder not used
        self.state_encoder = nn.Sequential(
            init_(nn.Conv2d(in_channels=global_input_channel, out_channels=global_out_channel1, kernel_size=global_kernel_size1, stride=global_stride1, padding=global_padding_size1)), 
            nn.AdaptiveMaxPool2d(24), self.active_func2, nn.Flatten(),init_(nn.Linear((global_out_channel1)*fixed_image_size, n_embd)), self.active_func2)
            # init_(nn.Linear(out_channel2, n_embd)), self.active_func2)



        # self.obs_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=input_channel, out_channels=out_channel1, kernel_size=kernel_size1, stride=stride1, padding=padding_size1)), self.active_func1,
        #     init_(nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=kernel_size2, stride=stride2, padding=padding_size2)), self.active_func1,
        #     nn.Flatten(), init_(nn.Linear((out_channel2)*conv_out_image_size2, n_embd), activate=True), self.active_func2)
            
        # This is using one layer of CNN and working well.
        # self.obs_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=input_channel, out_channels=out_channel1, kernel_size=kernel_size1, stride=stride1, padding=padding_size1)), 
        #     self.active_func1,  nn.Flatten(),init_(nn.Linear((out_channel1)*conv_out_image_size1, n_embd)), self.active_func2)
        #     # init_(nn.Linear(out_channel2, n_embd)), self.active_func2)
        

        # self.state_encoder = nn.Sequential(
        #     init_(nn.Conv2d(in_channels=input_channel, out_channels=out_channel1, kernel_size=kernel_size, stride=stride, padding=padding_size)), 
        #     self.active_func1, nn.Flatten(),init_(nn.Linear((out_channel1)*conv_out_image_size, n_embd)), self.active_func2)
        

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs, active_agents=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state: #not implemented
            state_embeddings = self.state_encoder(state)
            state_embeddings = state_embeddings.reshape(-1, self.n_agent, self.n_embd) # (n_rollout_threads, n_agent, n_embd)
            x = state_embeddings
        else:
            global_obs_encoding = self.global_obs_encoder(obs['global_agent_map'])
            local_obs_encoding = self.local_obs_encoder(obs['local_agent_map'])
            obs_encoding = torch.cat((global_obs_encoding, local_obs_encoding), dim=1) #concatenate local and global obs encodings
            obs_embeddings = self.obs_encoder(obs_encoding)
            agent_class_encoding = self.agent_class_encoder(obs['agent_class_identifier'])
            obs_embeddings = obs_embeddings + agent_class_encoding # (n_rollout_threads*n_agent, n_embd)
            obs_embeddings = obs_embeddings.reshape(-1, self.n_agent, self.n_embd) # (n_rollout_threads, n_agent, n_embd)
            x = obs_embeddings
        # print("shape of x is ", x.shape)
        # print("active agents are: ", active_agents)
        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)
        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_embd_vit, n_head, n_agent, n_agent_types,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type
        self.n_agent = n_agent


        input_channel = 1
        kernel_size = 3
        stride = 1
        padding_size = 1
        self.active_func1 = nn.ReLU()
        self.active_func2 = nn.GELU()

        """if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))""" #used for hands env (i think)

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    nn.GELU())
                self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))
                self.agent_class_encoder = nn.Sequential(init_(nn.Linear(n_agent_types, n_embd), activate=True), self.active_func2)

            elif action_type == 'MultiDiscrete':
                # self.action_encoder = nn.Sequential(nn.Flatten(), init_(nn.Linear(1*(action_dim[0]+1)*(action_dim[1]+1), n_embd-n_agent_types)), self.active_func2)
                self.action_encoder = nn.Sequential(init_(nn.Linear(1 + action_dim[0]+action_dim[1], n_embd, bias=False), activate=True), self.active_func2)
                self.agent_class_encoder = nn.Sequential(init_(nn.Linear(n_agent_types, n_embd), activate=True), self.active_func2)
                self.heads = []
                for act_dim in self.action_dim:
                    self.heads.append(nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                                    init_(nn.Linear(n_embd, act_dim))))
                self.heads = nn.ModuleList(self.heads)
                
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head) for _ in range(n_block)])
            

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs, rnn_states=None):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            if self.action_type == 'Discrete':
                action = action.reshape(-1, 1 + self.action_dim)
                action_encodings = self.action_encoder(action)

                agent_class_encoding = self.agent_class_encoder(obs['agent_class_identifier'])
                action_embeddings = action_encodings + agent_class_encoding
                action_embeddings = action_embeddings.reshape(-1, self.n_agent, self.n_embd) # (batch_size, n_agent, n_embd)
                x = self.ln(action_embeddings)
                for block in self.blocks:
                    x = block(x, obs_rep)
                logit = self.head(x)
            else:
                action = action.reshape(-1, 1, 1 + self.action_dim[0] + self.action_dim[1]) # (batch_size * n_agent, 1, act_dim[0]+1,act_dim[1]+1)

                action = torch.squeeze(action)
                action_encodings = self.action_encoder(action)
                agent_class_encoding = self.agent_class_encoder(obs['agent_class_identifier']) #not sure if this is necessary
                # action_embeddings = torch.cat((obs['agent_class_identifier'], action_encodings), dim=1) # (batch_size * n_agent, n_embd)
                action_embeddings = action_encodings + agent_class_encoding
                # action_encodings = action_encodings.reshape(-1, self.n_agent, self.n_embd) # (batch_size, n_agent, n_embd)
                action_embeddings = action_embeddings.reshape(-1, self.n_agent, self.n_embd) # (batch_size, n_agent, n_embd)
                # x = self.ln(action_encodings)
                x = self.ln(action_embeddings)
                for block in self.blocks:
                    x = block(x, obs_rep)
                logit = [] # logits
                for head in self.heads:
                    logit.append(head(x))      
                # logit = self.head(x)
        return logit, rnn_states


class MultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent, n_agent_types,
                 n_block, n_embd, n_embd_vit, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        # state unused
        # state_dim = 37
        if 'Dict' in obs_dim.__class__.__name__:
            self._mixed_obs = True
        else:
            self._mixed_obs = False


        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_embd_vit, n_head, n_agent, n_agent_types, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_embd_vit, n_head, n_agent, n_agent_types,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None, active_agents=None, rnn_states_actor=None, rnn_states_critic=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
                # state[key] = check(state[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        state = check(state).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        if rnn_states_actor is not None:
            rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        if rnn_states_critic is not None:
            rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_agents is not None:
            active_agents = check(active_agents).to(**self.tpdv)

        batch_size = np.shape(obs['agent_class_identifier'])[0]//self.n_agent

        v_loc, obs_rep = self.encoder(state, obs, active_agents)
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions, rnn_states_actor, active_agents)
        elif self.action_type == "MultiDiscrete":
            action = action.long()
            action_log, entropy = multidiscrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions, rnn_states_actor, active_agents) #action_log and entropy are lists
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False, active_agents=None, rnn_states_actor=None, rnn_states_critic=None):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)


        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
                # state[key] = check(state[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        state = check(state).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        

        batch_size = np.shape(obs['agent_class_identifier'])[0]//self.n_agent
        v_loc, obs_rep = self.encoder(state, obs, active_agents)
        if self.action_type == "Discrete":
            output_action, output_action_log, _ = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, rnn_states_actor, deterministic, active_agents)
        elif self.action_type == "MultiDiscrete":
            output_action, output_action_log, _ = multidiscrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, rnn_states_actor, deterministic, active_agents)
        else:
            output_action, output_action_log, _ = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)
        if rnn_states_actor is not None:
            return output_action, output_action_log, v_loc, rnn_states_actor, rnn_states_critic
        else:
            return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        if self._mixed_obs:
            for key in obs.keys():
                obs[key] = check(obs[key]).to(**self.tpdv)
                # state[key] = check(state[key]).to(**self.tpdv)
        else:
            obs = check(obs).to(**self.tpdv)
        state = check(state).to(**self.tpdv)

        v_tot, obs_rep = self.encoder(state, obs, active_agents=None)
        return v_tot



