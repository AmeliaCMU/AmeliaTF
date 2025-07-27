import torch
import torch.nn as nn

from easydict import EasyDict
from typing import Any, Tuple

from amelia_tf.models.components.gmm import GMM


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

        # Unpack ego agent information
        ego_agent_id = kwargs.get("ego_agent")
        ego_agent_id = torch.from_numpy(ego_agent_id)
        ego_agent_id = ego_agent_id.to(device).long()  # (B,)
        ego_index = ego_agent_id.view(B, 1, 1, 1).expand(-1, 1, self.hist_len, D)

        # Get ego agent history
        ego_hist = torch.gather(x[:, :, :self.hist_len, :], dim=1, index=ego_index)  # (B, 1, hist_len, D)
        ego_hist = ego_hist.squeeze(1)  # (B, hist_len, D)

        # Velocity
        timesteps = torch.arange(1, pred_len + 1, device=device).float()
        timesteps = timesteps.view(1, -1, 1).expand(B, -1, 3)
        deltas = ego_hist[:, -2, :3] - ego_hist[:, -1, :3]  # Speed components in X, Y, Z
        displacements = deltas.unsqueeze(1) * timesteps

        # Get initial values and components
        last_pos = ego_hist[:, -1, :3]

        # Create a dummy vector to populate with predictions
        future = torch.zeros(B, T, 3, device=device)
        # breakpoint()
        future[:, self.hist_len:, :] = last_pos.unsqueeze(1) + displacements

        # For usability with other evaluation repeat the prediction. Repeat prediction along H -> B,A H, T ,D
        H = self.decoder_head.num_futures
        repeated_future = future.unsqueeze(1).expand(-1, H, -1, -1)  # (B, H, T, D)

        # Replace
        pred_traj = torch.zeros(B, A, H, T, 3, device=device)
        ego_index_scatter = ego_agent_id.view(B, 1, 1, 1, 1).expand(-1, 1, H, T, 3)
        pred_traj.scatter_(1, ego_index_scatter, repeated_future.unsqueeze(1))

        # Reorder to be B, A, T, H, D
        pred_traj = pred_traj.permute(0, 1, 3, 2, 4)  # (B, A, T, H, D)

        # Get pred scores and sigma
        pred_scores = torch.ones(B, A, T, H)
        sigma = torch.zeros(B, A, T, H, 3)

        # future[:, :, 1] = last_pos[:, 1:2] + delta_y
        # future[:, :, 2] = last_pos[:, 2:3].expand(-1, pred_len)
        # future[:, :, 3] = last_heading.unsqueeze(1).expand(-1, pred_len)
        # last_heading = ego_hist[:, -1, 3]
        # heading_rad = torch.deg2rad(last_heading)
        # dir_x = torch.cos(heading_rad)
        # dir_y = torch.sin(heading_rad)

        # Calculate vectorized displacements
        # displacements = speed.view(B, 1) * timesteps
        # delta_x = displacements * dir_x.view(B, 1)
        # delta_y = displacements * dir_y.view(B, 1)
        return pred_scores, pred_traj, sigma
