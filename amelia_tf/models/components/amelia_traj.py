import torch
import torch.nn as nn

from easydict import EasyDict
from typing import Any, Tuple

from amelia_tf.models.components.self_attention import SelfAttentionBlock
from amelia_tf.models.components.gmm import GMM
from amelia_tf.models.components.common import MLP, LayerNorm

class AmeliaTraj(nn.Module):
    """ Base model for trajectory prediction on airport data. Baseline designed for trajectory data
    only. Largely based on the SceneTransformer: https://arxiv.org/pdf/2106.08417.pdf """
    def __init__(self, config: EasyDict) -> None:
        """ Class initialization. Builds the require components for the model.

        Input
        -----
            config[EasyDict]: a dictionary containing the parameters needed to build the model.
        """
        super().__init__()

        self.encoder_config = config.encoder
        self.decoder_config = config.decoder

        self.in_size = self.encoder_config.in_size + self.encoder_config.interp_flag
        self.embed_size = self.encoder_config.embed_size

        # Agent feature extraction
        self.agents_fe = nn.Sequential(*[
            nn.Linear(self.in_size, self.embed_size),
            LayerNorm(self.embed_size, self.encoder_config.bias),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size)
        ])

        # Positional encodings
        self.hist_len = self.encoder_config.hist_len
        self.pred_lens = self.encoder_config.pred_lens
        self.encoder_config.T_size = self.encoder_config.hist_len + max(self.pred_lens)
        self.time_pe = nn.Embedding(self.encoder_config.T_size, self.embed_size)

        self.drop = nn.Dropout(self.encoder_config.dropout)

        # Social-Temporal encoding. Largely based on SceneTransformer (https://arxiv.org/pdf/2106.08417.pdf)
        # which sequentially adds a transformer block across agents after a block across time.
        self.encoder_config.in_size = self.embed_size
        self.pre_attention_blocks = []
        for _ in range(self.encoder_config.num_blocks):
            self.pre_attention_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            self.pre_attention_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))
        self.pre_attention_blocks = nn.ModuleList(self.pre_attention_blocks)

        # Removes the artificial agent and timestep and refines the features.
        self.refine_mlp = MLP(self.encoder_config)

        # Finally, trajectory decoding is done using Gaussian Mixtures
        self.decoder_config.in_size = self.embed_size
        self.decoder_head = GMM(self.decoder_config)

        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    @property
    def num_dec_heads(self) -> int:
        return self.decoder_head.num_futures

    def get_num_params(self, non_embedding: bool = True) -> int:
        """ Returns the number of parameters in the model. For non-embedding count (default), the
        position embeddings get subtracted. The token embeddings would too, except due to the parameter
        sharing these params are actually used as weights in the final layer, so we include them. """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.time_pe.weight.numel()
        return n_params

    def _init_weights(self, module: Any) -> None:
        """ Weight initialization. """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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

        Outputs
        -------
            pred_scores[torch.tensor(B, A, T, H)]: prediction scores for each prediction trajectory's
                prediction head.
                H: number of predicted heads.
            mu[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
            sigma[torch.tensor(B, A, T, H, D)]: predicted sigmas for each trajectory.
        """
        device = x.device
        B, A, T, D = x.size()
        assert T <= self.encoder_config.T_size, \
            f"Can't forward sequence of length {T}, time block size is {self.encoder_config.T_size}"

        # x: (B, A, T, D=3) -> (B, A, T, D=embed_size)
        x = self.agents_fe(x)

        time_pe = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, T)
        # time emb: (1, 1, T, D=n_embed)
        time_emb = self.time_pe(time_pe).unsqueeze(dim=0)

        x = self.drop(x + time_emb)# + agents_emb)

        mask = kwargs.get('mask')
        # transformer blocks modeling interactions across-time and across-agents
        # x: (B, A, T, D=embed_size)
        for block in self.pre_attention_blocks:
            x = block(x, mask=mask) # output: (A, T, D=n_embed)

        x = self.refine_mlp(x)

        # decoding
        pred_scores, mu, sigma = self.decoder_head(x)
        return pred_scores, mu, sigma