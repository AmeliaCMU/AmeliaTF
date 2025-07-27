import torch
import torch.nn as nn

from easydict import EasyDict
from typing import Any, Tuple

from amelia_tf.models.components.gmm import GMM
from amelia_tf.utils.utils import separate_ego_agent


class ConstVelocity(nn.Module):
    """ Context-aware model for trajectory prediction on airport data. Baseline designed for both,
    trajectory and context data. Largely based on the SceneTransformer:
    https://arxiv.org/pdf/2106.08417.pdf """

    def __init__(self, config: EasyDict) -> None:
        super().__init__()

        self.encoder_config = config.encoder
        self.decoder_config = config.decoder

        self.in_size = self.encoder_config.in_size + self.encoder_config.interp_flag
        self.embed_size = self.encoder_config.embed_size

        # Positional encodings
        self.hist_len = self.encoder_config.hist_len
        self.pred_lens = self.encoder_config.pred_lens

        self.decoder_config.in_size = self.embed_size
        self.decoder_head = GMM(self.decoder_config)

    @property
    def num_dec_heads(self) -> int:
        return self.decoder_head.num_futures

    # def get_num_params(self, non_embedding: bool = True) -> int:
    #     """ Returns the number of parameters in the model. For non-embedding count (default), the
    #     position embeddings get subtracted. The token embeddings would too, except due to the parameter
    #     sharing these params are actually used as weights in the final layer, so we include them. """
    #     n_params = sum(p.numel() for p in self.parameters())
    #     if non_embedding:
    #         n_params -= self.time_pe.weight.numel()
    #     return n_params

    # def _init_weights(self, module: Any) -> None:
    #     """ Weight initialization. """
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.tensor, **kwargs) -> Tuple:
        """ Model's forward module.

        Inputs
        ------
            x[torch.tensor(B, A, T, D)]: input tensor containing the trajectory information.
                B: batch size
                A: number of agents
                T: trajectory length
                D: number of input dimensions.
            kwargs[Any]: other keyword arguments.
                Should contain a key 'context' containing the map information in vectorized format.
                c[torch.tensor(B, T, P, Dc)]: is the tensor containing the context information
                    P: number of polylines
                    Dc: number of input dimensions of the context.

        Outputs
        -------
            pred_scores[torch.tensor(B, A, T, H)]: prediction scores for each prediction trajectory's
                prediction head.
                H: number of predicted heads.
            mu[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
            sigma[torch.tensor(B, A, T, H, D)]: predicted sigmas for each trajectory.
        """

        device = x.device
        B, A, T, D = x.shape
        pred_len = max(self.pred_lens)
        H = self.decoder_head.num_futures
        
        # Init tensors to populate with predictions
        pred_scores = torch.ones(B, A, H)
        sigma = torch.zeros(B, A, T, H, 3)
        pred_traj = torch.zeros(B, A, T, 3)

        # Unpack ego agent information
        ego_id = kwargs.get("ego_agent")
        ego_agent_id = torch.from_numpy(ego_id)
        ego_index = ego_agent_id.view(B, 1, 1, 1).expand(-1, 1, self.hist_len, D)

        # Get ego agent's history and calculate velocity at last timestep
        ego_hist = torch.gather(x, dim=1, index=ego_index).squeeze(1)
        pos_t1 = ego_hist[:, -1, :3]  
        pos_t0 = ego_hist[:, -2, :3]  
        velocity = pos_t1 - pos_t0    # Velocity for  xyz
        
        # Propagate velocity by aggregating across timesteps
        timesteps = torch.arange(1, pred_len + 1, device=device).view(1, pred_len, 1)   # (1, T_pred, 1)
        displacements = velocity.unsqueeze(1) * timesteps
        ego_future = pos_t1.unsqueeze(1) + displacements

        ego_traj = torch.cat([ego_hist[:, :, :3], ego_future], dim=1)  # (B, T, 3)
        
        # Populate the prediction tensors
        src = ego_traj.unsqueeze(1)                                   # (B, 1, T_total, 3)
        ego_idx = ego_agent_id.view(B, 1, 1, 1).expand(-1, 1, T, 3)  # (B, 1, T_total, 3)
        pred_traj.scatter_(dim=1, index=ego_idx, src=src)             # (B, A, T_total, 3)

        # Repeat trajectory H times for compatibility with GMM output
        
        pred_traj = pred_traj.unsqueeze(3)  # (B, A, T_total, 1, 3)
        # Step 2: repeat H times along head dim
        pred_traj = pred_traj.expand(-1, -1, -1, H, -1)  # (B, A, T_total, H, 3)
        
        #Sanity check, make sure that the placed traj is the same as the ego agent's
        _traj = separate_ego_agent(pred_traj, ego_agent_id)
        _traj = _traj[:, :, :, 0, ].squeeze(1)
        if not torch.allclose(_traj, ego_traj):        
            raise ValueError("Ego agent trajectory does not match the expected trajectory.")
        
        return pred_scores, pred_traj, sigma
