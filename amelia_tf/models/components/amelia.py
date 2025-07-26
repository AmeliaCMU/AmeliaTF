import torch
import torch.nn as nn


from easydict import EasyDict
from omegaconf import OmegaConf
from typing import Any, Tuple

import os
import json
from huggingface_hub import PyTorchModelHubMixin

from amelia_tf.models.components.self_attention import SelfAttentionBlock
from amelia_tf.models.components.cross_attention import CrossAttentionBlock
from amelia_tf.models.components.gmm import GMM
from amelia_tf.models.components.common import MLP, LayerNorm


class AmeliaTF(nn.Module, PyTorchModelHubMixin, repo_url="https://huggingface.co/AmeliaCMU/AmeliaTF-Seen-7"):
    """ Context-aware model for trajectory prediction on airport data. Baseline designed for both,
    trajectory and context data. Largely based on the SceneTransformer:
    https://arxiv.org/pdf/2106.08417.pdf """

    def __init__(self, config: EasyDict) -> None:
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

        # Context feature extraction
        # TODO: once the best configuration is found, remove this mess.
        ctx_enc_type = self.encoder_config.context_encoder_type
        if ctx_enc_type == 'v0':
            from amelia_tf.models.components.context import ContextNetv0 as ContextNet
            context_config = self.encoder_config.contextnet_v0
        elif ctx_enc_type == 'v1':
            from amelia_tf.models.components.context import ContextNetv1 as ContextNet
            context_config = self.encoder_config.contextnet_v1
        elif ctx_enc_type == 'v2':
            from amelia_tf.models.components.context import ContextNetv2 as ContextNet
            context_config = self.encoder_config.contextnet_v2
        elif ctx_enc_type == 'v3':
            from amelia_tf.models.components.context import ContextNetv3 as ContextNet
            context_config = self.encoder_config.contextnet_v3
        elif ctx_enc_type == 'v4':
            from amelia_tf.models.components.context import ContextNetv4 as ContextNet
            context_config = self.encoder_config.contextnet_v4
        else:
            raise NotImplementedError
        self.context_fe = ContextNet(context_config)

        # Positional encodings
        self.hist_len = self.encoder_config.hist_len
        self.pred_lens = self.encoder_config.pred_lens
        self.time_pe = nn.Embedding(self.encoder_config.T_size, self.embed_size)

        self.drop = nn.Dropout(self.encoder_config.dropout)

        # Social-Temporal encoding. Largely based on SceneTransformer (https://arxiv.org/pdf/2106.08417.pdf)
        # which sequentially adds a transformer block across agents after a block across time.
        self.encoder_config.in_size = self.embed_size
        self.att_blocks = []
        for _ in range(self.encoder_config.num_satt_pre_blocks):
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))

        # Cross-attention block for agents and map
        for _ in range(self.encoder_config.num_catt_pre_blocks):
            self.att_blocks.append(CrossAttentionBlock(self.encoder_config))

        # Intermediate self-attention and cross-attention blocks
        for _ in range(self.encoder_config.num_satt_blocks):
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))

        for _ in range(self.encoder_config.num_catt_blocks):
            self.att_blocks.append(CrossAttentionBlock(self.encoder_config))

        # Post self-attention blocks
        for _ in range(self.encoder_config.num_satt_post_blocks):
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))
        self.att_blocks = nn.ModuleList(self.att_blocks)

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
        B, A, T, D = x.size()
        assert T <= self.encoder_config.T_size, \
            f"Can't forward sequence of length {T}, time block size is {self.encoder_config.T_size}"

        # x: (B, A, T, Dx=3) -> (B, A, T, D=embed_size)
        x = self.agents_fe(x)

        time_pe = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, T)
        # time emb: (1, 1, T, D=n_embed)
        time_emb = self.time_pe(time_pe).unsqueeze(dim=0)
        x = self.drop(x + time_emb)

        # c: (B, T, P, Dc=5)
        cx = kwargs.get('context')
        adj = kwargs.get('adjacency')
        assert not cx is None

        # cx: (B * A, D=embed_size)
        cx = self.context_fe(cx, adj=adj)

        # transformer blocks modeling interactions across-time and across-agents
        mask = kwargs.get('mask')
        for block in self.att_blocks:
            x = block(x, cx, mask=mask)

        # prediction
        x = self.refine_mlp(x)

        # decoding
        pred_scores, mu, sigma = self.decoder_head(x)
        return pred_scores, mu, sigma

    def save_pretrained(self, save_directory, **kwargs):
        save_directory = os.path.abspath(save_directory)
        os.makedirs(save_directory, exist_ok=True)
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        # Convert ListConfig to dict or list
        encoder_config_dict = OmegaConf.to_container(self.encoder_config, resolve=True)
        decoder_config_dict = OmegaConf.to_container(self.decoder_config, resolve=True)
        config_dict = {"encoder": encoder_config_dict, "decoder": decoder_config_dict,
                       "kwargs": kwargs}

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f)
        print(f"Model and config saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory, device="cpu", **kwargs):
        # Load config
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config_dict = json.load(f)
        # Convert config_dict to EasyDict or your config format
        config = EasyDict(config_dict)
        # Initialize the model with the loaded config
        model = cls(config, **kwargs)
        # Load weights
        state_dict = torch.load(os.path.join(load_directory, "pytorch_model.bin"), map_location=device)
        breakpoint()
        model.load_state_dict(state_dict)
        return model
