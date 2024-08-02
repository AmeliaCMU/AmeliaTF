import torch    
import torch.nn as nn

from easydict import EasyDict
from torch.nn import functional as F

from typing import Tuple

class GMM(nn.Module):
    """ Gaussian Mixture Model module. """
    def __init__(self, config: EasyDict) -> None:
        """ Self-Attention intialization method. 
        
        Inputs
        ------
            config[EasyDict]: dictionary with configuration parameters. 
        """
        super().__init__()
        self.config = config

        self.num_futures = config.num_futures
        gmm_embd = config.in_size // config.num_futures

        self.future_heads = nn.Sequential(
            nn.Linear(gmm_embd, 4 * gmm_embd),
            nn.GELU(),
            nn.Linear(4 * gmm_embd, config.num_dims, bias=False)                                    
        )
        self.out_dim = int((config.num_dims-1)//2)

    def forward(self, x: torch.tensor) -> Tuple:
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
            pred_scores[torch.tensor(B, A, T, H)]: prediction scores for each prediction trajectory's
                prediction head. 
                H: number of predicted heads. 
            mu[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory. 
            sigma[torch.tensor(B, A, T, H, D)]: predicted sigmas for each trajectory.  
        """
        B, A, T, C = x.size() 
        x = x.view(B, A, T, self.num_futures, C // self.num_futures)
        
        out = self.future_heads(x)

        pred_scores = F.softmax(out[...,-1].mean(-2),dim=-1) ##last dim is mixture probablity 
        mu = out[..., :self.out_dim]
        sigma = torch.exp(out[..., self.out_dim:(self.out_dim * 2)])
        
        return pred_scores, mu, sigma