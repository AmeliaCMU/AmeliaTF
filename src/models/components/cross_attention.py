import math
import torch
import torch.nn as nn

from easydict import EasyDict
from torch.nn import functional as F

from src.models.components.common import MLP, LayerNorm

class CrossAttentionBlock(nn.Module):
    """ Cross-Attention Block module. """
    def __init__(self, config: EasyDict) -> None:
        """ Self-Attention intialization method. 
        
        Inputs
        ------
            config[EasyDict]: dictionary with configuration parameters. 
        """
        super().__init__()
        assert config.embed_size % config.num_heads == 0

        self.num_heads = config.num_heads
        self.embed_size = config.embed_size
        self.dropout = config.dropout

        # architecture
        self.ln_1 = LayerNorm(config.embed_size, bias=config.bias)
        self.q = nn.Linear(config.embed_size, config.embed_size, bias=config.bias)
        self.kv = nn.Linear(config.embed_size, 2 * config.embed_size, bias=config.bias)
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
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(config.T_size, config.T_size))
                .view(1, 1, 1, config.T_size, config.T_size))
        
        scene_mask = torch.zeros(config.T_size)
        scene_mask[:config.hist_len] = 1
        self.register_buffer("scene_mask", scene_mask.view(-1, 1, 1, 1))

    def attn(self, x: torch.tensor, cx: torch.tensor, mask = None) -> torch.tensor:
        """ Performs Multi-Head Cross-Attention (MHA). 
        
        Input
        -----
            x[torch.tensor(B, A, T, D)]: input tensor used as keys (K) and values (V).
                B: batch size
                A: number of agents
                T: trajectory length
                D: embedding size
            cx[torch.tensor(B, T, D)]: input tensor used as queries (Q).

        Output
        ------
            y[torch.tensor(B, A, T, D)]: cross-attended output. 
        """
        B, A, T, D = x.size()
        
        # Q: (B, A, D) -> (B, A, 1, D) -> (B, A, T, D)
        Q = self.q(x)
        # K, V: (B, A, T, D)
        K, V = self.kv(cx)[..., None, :].repeat(1, 1, T, 1).split(self.embed_size, dim=3)

        # Q, K, V: (B, A, H, T, HD)
        K = K.view(B, A, T, self.num_heads, D // self.num_heads).transpose(2, 3) 
        Q = Q.view(B, A, T, self.num_heads, D // self.num_heads).transpose(2, 3) 
        V = V.view(B, A, T, self.num_heads, D // self.num_heads).transpose(2, 3)

        # att: (B, A, H, T, T)                                        
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))           
        # causal masking 
        att = att.masked_fill(self.causal_mask[:, :, :, :T, :T] == 0, float('-inf')) 
        att = F.softmax(att, dim=-1)                                             
        att = self.attn_dropout(att)                                              

        # re-assemble all head outputs side by side 
        # y: (B, A, H, T, HD)
        y = att @ V 
        y = y.transpose(2, 3).contiguous().view(B, A, T, D) # (B, A, T, D=embed_size)
        
        y = self.c_proj(y)          # (A, T, D=embed_size)
        y = self.resid_dropout(y)   # (A, T, D=embed_size)
        return y

    def forward(self, x: torch.tensor, cx: torch.tensor, mask = None) -> torch.tensor:
        """ Model's forward function. 
        
        Input
        -----
            x[torch.tensor]: input tensor to be self-attended. 

        Output
        ------
            x[torch.tensor]: attended output tensor. 
        """
        x = self.ln_1(x)
        x = x + self.attn(x, cx, mask=mask)
        x = self.ln_2(x)
        x = x + self.mlp(x) 
        return x