
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

from attentions import MultiHeadSelfAttention, FFN
from activations import get_activation
from transformer_utils import *
# from torchmetrics.functional import pairwise_cosine_similarity

from transformers.models.bert.modeling_bert import BertEncoder, BaseModelOutputWithPoolingAndCrossAttentions, \
    apply_chunking_to_forward, BertAttention, BertIntermediate, BertOutput, Optional, Tuple, Union, BaseModelOutputWithPastAndCrossAttentions

from typing import Dict, List, Optional, Set, Tuple, Union



class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)
        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

class HyperAdapterNetworks(nn.Module):
    def __init__(self, config, task_dim=8, rank=16, con_dim=64):
        super().__init__()
        self.config = config
        self.in_dim = config.hidden_size
        self.task_dim = task_dim
        self.con_dim = con_dim
        self.rank = rank
        self.n_layers = config.num_hidden_layers
        self.adapter_dist = None
        
        self.hyper_lower = torch.nn.Parameter(torch.randn(task_dim * rank, con_dim))
        nn.init.normal_(self.hyper_lower, std=1e-2)
        self.hyper_lower.requires_grad = True

        self.hyper_higher = torch.nn.Parameter(torch.randn(con_dim, config.hidden_size))
        nn.init.normal_(self.hyper_higher, std=1e-2)
        self.hyper_higher.requires_grad = True

        input_token_dim = int(task_dim/1)
        self.layer_tokens = nn.Embedding(config.num_hidden_layers, input_token_dim) # (6, 8)
        self.dist_tokens = nn.Linear(config.num_labels, input_token_dim) # (10, 8)
        self.input_tokens = nn.Linear(config.hidden_size, input_token_dim)
        
        self.adapter_bias = torch.nn.Parameter(torch.randn(1, rank * config.hidden_size))
        nn.init.zeros_(self.adapter_bias)
        self.adapter_bias.requires_grad = True

        self.dist_stats = None # (1, 10)

        self.bias1 = torch.nn.Parameter(torch.randn(1, 1, rank))
        self.bias2 = torch.nn.Parameter(torch.randn(1, 1, config.hidden_size))
        nn.init.zeros_(self.bias1)
        nn.init.zeros_(self.bias2)
        self.bias1.requires_grad = True
        self.bias2.requires_grad = True

    def forward(self, inputs, layer_idx): 
        
        # 1. distribution features
        dist = self.dist_tokens(self.dist_stats)
        
        # 2. contextual features
        batched_mean_feat = inputs / torch.norm(inputs, dim=-1, keepdim=True)
        # batched_mean_feat = inputs.max(dim=0, keepdim = True)[0] #(1, dim)
        batched_mean_feat = batched_mean_feat.mean(dim=0, keepdim=True)
        batched_mean_feat = self.input_tokens(batched_mean_feat) #(1,task)

        # 3. layer features
        layer = self.layer_tokens(torch.tensor([layer_idx]).long().cuda())

        input_tokens = dist + layer + batched_mean_feat 
        
        # HyperNetworks (Cross-Lingual, Heterogeneity -> Client Drift)
        adapter_weights = (self.hyper_lower @ self.hyper_higher).view(self.task_dim, self.in_dim * self.rank) + self.adapter_bias
        adapter_weights = input_tokens @ (adapter_weights / adapter_weights.norm(dim=-1, keepdim=True)) 

        up_weights = adapter_weights.view(1, self.in_dim, self.rank) 
        down_weights = up_weights.view(1, self.rank, self.in_dim) 
        return up_weights, down_weights, self.bias1, self.bias2


class HyperNetworkTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.n_heads == 0
        self.gelu = nn.GELU()
        
        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
    
        self.q_out1 = None
        self.q_out2 = None

    def base_adapter_function(self, w_up, w_down, b1, b2, x):
        hidden = self.gelu(x @ w_up + b1)
        out = (hidden @ w_down) + b2
        out = out + x
        return out, hidden

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        hyper_nets=None,
        layer_idx=-1,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        
        up_weights1, down_weights1, bias1, bias2 = hyper_nets(sa_output.mean(dim=1), layer_idx)
        sa_output, self.q_out1 = self.base_adapter_function(up_weights1, down_weights1, bias1, bias2, sa_output)
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output) 

        up_weights2, down_weights2, bias1, bias2 = hyper_nets(ffn_output.mean(dim=1), layer_idx)
        ffn_output, self.q_out2 = self.base_adapter_function(up_weights2, down_weights2, bias1, bias2, ffn_output)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output
