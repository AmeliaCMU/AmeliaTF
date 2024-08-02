import math
import torch
import torch.nn as nn

from easydict import EasyDict
from torch.nn import functional as F

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x: torch.tensor) -> torch.tensor:
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MLP(nn.Module):
    """ Multilayer Perceptron (MLP) module. """
    def __init__(self, config: EasyDict) -> None:
        """ MLP module. 
        
        Inputs
        ------
            config[EasyDict]: dictionary with configuration parameters. 
        """
        super().__init__()

        self.c_fc    = nn.Linear(config.embed_size, 4 * config.embed_size, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.embed_size, config.embed_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ Model's forward function. 
        
        Input
        -----
            x[torch.tensor(B, A, T, C)]: input tensor to be decoded. 
                B: batch size
                A: number of agents
                T: trajectory length
                C: number of input dimensions. 

        Outputs
        -------
            x[torch.tensor(B, A, T, H)]: extracted feature. 
        """
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim: int, bias: bool) -> None:
        """ LayerNorm initialization. 
        
        Inputs
        ------
            ndim[int]: number of input dimensions.
            bias[bool]: whether to add a bias.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.tensor) -> torch.tensor:
        """ Model's forward function. 
        
        Input
        -----
            x[torch.tensor(B, A, T, C)]: input tensor to be decoded. 
                B: batch size
                A: number of agents
                T: trajectory length
                C: number of input dimensions. 

        Outputs
        -------
            x[torch.tensor(B, A, T, H)]: extracted feature. 
        """
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)