import math
import torch
import torch.nn as nn

from easydict import EasyDict
from torch.nn import functional as F

from src.models.components.common import MLP, LayerNorm

class BaseSelfAttentionBlock(nn.Module):
    """ Self-Attention Block module. """
    def __init__(self, config: EasyDict) -> None:
        """ Self-Attention intialization method. 
        
        Inputs
        ------
            config[EasyDict]: dictionary with configuration parameters. 
            across[str]: can be either 'time' or 'agents' and is used to specify whether the attention
                operations are performed across the time or agents dimentions. 
        """
        super().__init__()
        assert config.embed_size % config.num_heads == 0
    
        self.num_heads = config.num_heads
        self.embed_size = config.embed_size
        self.dropout = config.dropout

        # architecture
        self.ln_1 = LayerNorm(config.embed_size, bias=config.bias)
        self.qkv = nn.Linear(config.embed_size, 3 * config.embed_size, bias=config.bias)
        self.c_proj = nn.Linear(config.embed_size, config.embed_size, bias=config.bias)
        self.ln_2 = LayerNorm(config.embed_size, bias=config.bias)
        self.mlp = MLP(config)

        # regularization 
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # TODO: fix this self.flash issue
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer(
        #         "bias", torch.tril(torch.ones(config.T_size, config.T_size))
        #             .view(1, 1, config.T_size, config.T_size))
        
    def attn(self, x: torch.tensor, cx = None) -> torch.tensor:
        """ Performs Multi-Head Attention (MHA). 
        
        Input
        -----
            x[torch.tensor(B, A, T, D)]: input tensor over which to compute MHA. 
                B: batch size
                A: number of agents
                T: trajectory length
                C: embedding size
        
        Output
        ------
            y[torch.tensor(B, A, T, D)]: attended output. 
        """
        B, A, T, C = x.size()
       
        # query, key, values for all heads in batch and move head forward to be the batch dim
        Q, K, V  = self.qkv(x).split(self.embed_size, dim=3)
        
        K = K.view(B, A, T, self.num_heads, C // self.num_heads).transpose(2, 3) # (B, A, H, T, HD)
        Q = Q.view(B, A, T, self.num_heads, C // self.num_heads).transpose(2, 3) # (B, A, H, T, HD)
        V = V.view(B, A, T, self.num_heads, C // self.num_heads).transpose(2, 3) # (B, A, H, T, HD)

        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))           # (B, A, H, T, T)                                        
        att = F.softmax(att, dim=-1)                                              # (B, A, H, T, T)
        att = self.attn_dropout(att)                                              # (B, A, H, T, T)
        y = att @ V                                                               # (B, A, H, T, HD)

        # re-assemble all head outputs side by side 
        y = y.transpose(2, 3).contiguous().view(B, A, T, C) # (B, A, T, D=embed_size)
       
        y = self.c_proj(y)          # (A, T, D=embed_size)
        y = self.resid_dropout(y)   # (A, T, D=embed_size)
        return y

    def forward(self, x: torch.tensor, cx = None, mask = None) -> torch.tensor:
        """ Model's forward function. 
        
        Input
        -----
            x[torch.tensor]: input tensor to be self-attended. 

        Output
        ------
            x[torch.tensor]: attended output tensor. 
        """
        x = self.ln_1(x)
        x = x + self.attn(x)
        x = self.ln_2(x)
        x = x + self.mlp(x)
        return x

class SelfAttentionBlock(nn.Module):
    """ Self-Attention Block module. """
    def __init__(self, config: EasyDict, across: str = 'time') -> None:
        """ Self-Attention intialization method. 
        
        Inputs
        ------
            config[EasyDict]: dictionary with configuration parameters. 
            across[str]: can be either 'time' or 'agents' and is used to specify whether the attention
                operations are performed across the time or agents dimentions. 
        """
        super().__init__()
        assert config.embed_size % config.num_heads == 0
        assert across in ['time', 'agents'], f"Across: {across} not in {['time', 'agents']}"

        self.across = across
        self.num_heads = config.num_heads
        self.embed_size = config.embed_size
        self.dropout = config.dropout

        # architecture
        self.ln_1 = LayerNorm(config.embed_size, bias=config.bias)
        self.qkv = nn.Linear(config.embed_size, 3 * config.embed_size, bias=config.bias)
        self.c_proj = nn.Linear(config.embed_size, config.embed_size, bias=config.bias)
        self.ln_2 = LayerNorm(config.embed_size, bias=config.bias)
        self.mlp = MLP(config)

        # regularization 
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(config.T_size, config.T_size))
                .view(1, 1, 1, config.T_size, config.T_size))
        
        self.tmask, self.amask = float('-inf'), -100
        scene_mask = torch.zeros(config.T_size)
        scene_mask[:config.hist_len] = 1
        self.register_buffer("scene_mask", scene_mask.view(-1, 1, 1, 1))

    def attn(self, x: torch.tensor, cx = None, mask = None) -> torch.tensor:
        """ Performs Multi-Head Attention (MHA). 
        
        Input
        -----
            x[torch.tensor(B, A, T, D)]: input tensor over which to compute MHA. 
                B: batch size
                A: number of agents
                T: trajectory length
                C: embedding size
        
        Output
        ------
            y[torch.tensor(B, A, T, D)]: attended output. 
        """
        B, A, T, C = x.size()

        # Query, key, values for all heads in batch and move head forward to be the batch dim
        Q, K, V  = self.qkv(x).split(self.embed_size, dim=3)

        if self.across == 'time':
            K = K.view(B, A, T, self.num_heads, C // self.num_heads).transpose(2, 3) # (B, A, H, T, HD)
            Q = Q.view(B, A, T, self.num_heads, C // self.num_heads).transpose(2, 3) # (B, A, H, T, HD)
            V = V.view(B, A, T, self.num_heads, C // self.num_heads).transpose(2, 3) # (B, A, H, T, HD)
            att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))          # (B, A, H, T, T)  

            # Apply masks                                      
            # Causal masking to avoid peeking into the future. 
            att = att.masked_fill(self.causal_mask[:, :, :, :T, :T] == 0, self.tmask) # (B, A, H, T, T)

            # Interpolation mask. Masks out data points that were interpolated. 
            # NOTE: Interpolated mask value == 0 means that the data point was interpolated. 
            # NOTE: Original mask size is (B, A, T). 
            # if not mask is None:
            #     mask_time = mask.unsqueeze(2).repeat(1, 1, T, 1).view(B, A, 1, T, T)
            #     att = att.masked_fill(mask_time == 1.0, self.tmask)

            att = F.softmax(att, dim=-1)  # (B, A, H, T, T)
            # if att.isnan().any():
            #     breakpoint()
            att = self.attn_dropout(att)  # (B, A, H, T, T)
            y = att @ V                   # (B, A, H, T, HD)
    
            # Re-assemble all head outputs side by side 
            y = y.transpose(2, 3).contiguous().view(B, A, T, C) # (B, A, T, D=embed_size)
        else:
            K = K.view(B, A, T, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1, 4) # (T, H, A, HD)
            Q = Q.view(B, A, T, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1, 4) # (T, H, A, HD)
            V = V.view(B, A, T, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1, 4) # (T, H, A, HD)

            att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1))) # (T, H, A, N)

            # if not mask is None:
            #     # Apply interpolation mask. Masks out data points that were interpolated. 
            #     # NOTE: Interpolated mask value == 0 means that the data point was interpolated. 
            #     # NOTE: Original mask size is (B, A, T).
            #     mask_agents = mask.permute(0, 2, 1).unsqueeze(2).repeat(1, 1, A, 1).view(B, T, 1, A, A)
            #     att = att.masked_fill(mask_agents == 1.0, self.amask)

            att = F.softmax(att, dim=-1) # (T, H, A, N)
            # if att.isnan().any():
            #     breakpoint()
            att = self.attn_dropout(att) # (T, H, A, N)
            y = att @ V                  # (T, H, A, N)

            # (T, nh, B, ns) -> (B, T, nh, ns) -> (B, T, C)
            y = y.permute(0, 3, 1, 2, 4).contiguous().view(B, A, T, C) # (B, A, T, D=embed_size)

        y = self.c_proj(y)          # (A, T, D=embed_size)
        y = self.resid_dropout(y)   # (A, T, D=embed_size)
        return y

    def forward(self, x: torch.tensor, cx = None, mask = None) -> torch.tensor:
        """ Model's forward function. 
        
        Input
        -----
            x[torch.tensor]: input tensor to be self-attended. 

        Output
        ------
            x[torch.tensor]: attended output tensor. 
        """
        x = self.ln_1(x)
        x = x + self.attn(x, mask=mask)
        x = self.ln_2(x)
        x = x + self.mlp(x)
        return x