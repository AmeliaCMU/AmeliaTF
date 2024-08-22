import torch
import torch.nn as nn

from easydict import EasyDict

from amelia_tf.models.components.self_attention import BaseSelfAttentionBlock

class ContextNetv0(nn.Module):
    """ Context module used to extract polyline features of the map. It simply consists of MLP + BN
    blocks and it does not implement feature aggregation. """
    def __init__(self, config: EasyDict) -> None:
        super().__init__()
        in_size = config.in_size
        embed_size = config.embed_size
        num_vectors = config.num_vectors

        self.pointnet = [
            nn.Linear(in_size, embed_size, bias=True),
            nn.BatchNorm1d(num_vectors),
            nn.ReLU()
        ]
        for _ in range(config.num_layers-1):
            self.pointnet.append(nn.Linear(embed_size, embed_size, bias=True))
            self.pointnet.append(nn.BatchNorm1d(num_vectors))
            self.pointnet.append(nn.ReLU())
        self.pointnet = nn.Sequential(*self.pointnet)

        self.pointnet_out = [
            nn.Linear(num_vectors * embed_size, num_vectors * (embed_size // 8), bias=True),
            nn.BatchNorm1d(num_vectors * (embed_size // 8)),
            nn.ReLU(),
            nn.Linear(num_vectors * (embed_size // 8), embed_size)
        ]
        self.pointnet_out = nn.Sequential(*self.pointnet_out)

    def forward(self, x, **kwargs):
        # B, N, P, D
        B, N, P, D = x.size()
        x = x.reshape(B * N, P, D)
        x = self.pointnet(x)

        BN, _, E = x.size()
        x = x.reshape(BN, -1)
        x = self.pointnet_out(x)
        x = x.reshape(B, N, E)
        return x

class ContextNetv1(nn.Module):
    """ Context module used to extract vector features of the map. It simply consists of MLP + BN
    blocks and it implements max-pooling as feature aggregation to reduce the number of polylines. """
    def __init__(self, config: EasyDict) -> None:
        super().__init__()
        in_size = config.in_size
        embed_size = config.embed_size
        num_vectors = config.num_vectors
        ker_size = config.ker_size
        assert num_vectors % ker_size == 0

        self.pointnet = [
            nn.Linear(in_size, embed_size, bias=True),
            nn.BatchNorm1d(num_vectors),
            nn.ReLU()
        ]
        for _ in range(config.num_layers-1):
            self.pointnet.append(nn.Linear(embed_size, embed_size, bias=True))
            self.pointnet.append(nn.BatchNorm1d(num_vectors))
            self.pointnet.append(nn.ReLU())
        self.pointnet = nn.Sequential(*self.pointnet)

        # striding only over polyline dimension, i.e., we want to keep the feature size the same,
        # just reduce the number of polylines
        # x: (B, A, P, D) -> (B, A, P//ker_size, D)
        self.pool = nn.MaxPool2d(kernel_size=(ker_size, 1))

        # Contextout
        ctx_in = num_vectors // ker_size
        self.pointnet_out = [
            nn.Linear(ctx_in * embed_size, ctx_in * (embed_size // 8), bias=True),
            nn.ReLU(),
            nn.Linear(ctx_in * (embed_size // 8), embed_size)
        ]
        self.pointnet_out = nn.Sequential(*self.pointnet_out)

    def forward(self, x, **kwargs):
        # B, N, P, D
        B, N, P, D = x.size()
        x = x.reshape(B * N, P, D)
        x = self.pointnet(x)

        # Vector feature extraction
        x = x.reshape(B, N, P, -1)
        x = self.pool(x)

        # Full context output feature with pseudo-aggregation (ideally, here, we'd do a maxpool
        # over labels or regions and then pass them to the MLP)
        x = x.reshape(B, N, -1)
        x = self.pointnet_out(x)
        return x

class ContextNetv2(nn.Module):
    """ Context module used to extract vector features of the map. It consists of MLP + BN + MHA
    blocks and with no feature aggregation. """
    def __init__(self, config: EasyDict) -> None:
        super().__init__()
        in_size = config.in_size
        embed_size = config.embed_size
        num_vectors = config.num_vectors

        self.pointnet = [
            nn.Linear(in_size, embed_size, bias=True),
            nn.BatchNorm1d(num_vectors),
            nn.ReLU()
        ]
        for _ in range(config.num_layers-1):
            self.pointnet.append(nn.Linear(embed_size, embed_size, bias=True))
            self.pointnet.append(nn.BatchNorm1d(num_vectors))
            self.pointnet.append(nn.ReLU())
        self.pointnet = nn.Sequential(*self.pointnet)

        # Multi-Head Attention (MHA) to learn interactions between polylines
        self.pointent_int = []
        for _ in range(config.num_satt_layers):
            self.pointent_int.append(BaseSelfAttentionBlock(config))
        self.pointent_int = nn.Sequential(*self.pointent_int)

        self.pointnet_out = [
            nn.Linear(num_vectors * embed_size, num_vectors * (embed_size // 8), bias=True),
            nn.ReLU(),
            nn.Linear(num_vectors * (embed_size // 8), embed_size)
        ]
        self.pointnet_out = nn.Sequential(*self.pointnet_out)

    def forward(self, x, **kwargs):
        # B, N, P, D
        # B, N, P, D
        B, N, P, D = x.size()
        x = x.reshape(B * N, P, D)
        x = self.pointnet(x)

        # Vector interaction / aggregation
        x = x.reshape(B, N, P, -1)

        # Vector feature extraction
        x = self.pointent_int(x)

        # Full context output feature with pseudo-aggregation (ideally, here, we'd do a maxpool
        # over labels or regions and then pass them to the MLP)
        x = x.reshape(B, N, -1)
        x = self.pointnet_out(x)
        return x

class ContextNetv3(nn.Module):
    """ Context module used to extract vector features of the map. It consists of MLP + BN + MHA
    blocks and with no feature aggregation. """
    def __init__(self, config: EasyDict) -> None:
        super().__init__()
        in_size = config.in_size
        embed_size = config.embed_size
        num_vectors = config.num_vectors
        ker_size = config.ker_size
        assert num_vectors % ker_size == 0

        self.pointnet = [
            nn.Linear(in_size, embed_size, bias=True),
            nn.BatchNorm1d(num_vectors),
            nn.ReLU()
        ]
        for _ in range(config.num_layers-1):
            self.pointnet.append(nn.Linear(embed_size, embed_size, bias=True))
            self.pointnet.append(nn.BatchNorm1d(num_vectors))
            self.pointnet.append(nn.ReLU())
        self.pointnet = nn.Sequential(*self.pointnet)

        self.pool = nn.MaxPool2d(kernel_size=(ker_size, 1))

        # Multi-Head Attention (MHA) to learn interactions between polylines
        self.pointent_int = []
        for _ in range(config.num_satt_layers):
            self.pointent_int.append(BaseSelfAttentionBlock(config))
        self.pointent_int = nn.Sequential(*self.pointent_int)

        ctx_in = num_vectors // ker_size
        self.pointnet_out = [
            nn.Linear(ctx_in * embed_size, ctx_in * (embed_size // 8), bias=True),
            nn.ReLU(),
            nn.Linear(ctx_in * (embed_size // 8), embed_size)
        ]
        self.pointnet_out = nn.Sequential(*self.pointnet_out)

    def forward(self, x, **kwargs):
        # B, N, P, D
        # B, N, P, D
        B, N, P, D = x.size()
        x = x.reshape(B * N, P, D)
        x = self.pointnet(x)

        # Vector interaction / aggregation
        x = x.reshape(B, N, P, -1)
        x = self.pool(x)
        x = self.pointent_int(x)

        # Full context output feature with pseudo-aggregation (ideally, here, we'd do a maxpool
        # over labels or regions and then pass them to the MLP)
        x = x.reshape(B, N, -1)
        x = self.pointnet_out(x)
        return x

class ContextNetv4(nn.Module):
    """ Context module used to extract vector features of the map. It consists of MLP + BN + MHA
    blocks and with no feature aggregation. """
    def __init__(self, config: EasyDict) -> None:
        super().__init__()
        in_size = config.in_size
        embed_size = config.embed_size
        num_vectors = config.num_vectors

        self.pointnet = [
            nn.Linear(in_size, embed_size, bias=True),
            nn.BatchNorm1d(num_vectors),
            nn.ReLU()
        ]
        for _ in range(config.num_layers-1):
            self.pointnet.append(nn.Linear(embed_size, embed_size, bias=True))
            self.pointnet.append(nn.BatchNorm1d(num_vectors))
            self.pointnet.append(nn.ReLU())
        self.pointnet = nn.Sequential(*self.pointnet)

        # Multi-Head Attention (MHA) to learn interactions between polylines
        self.pointent_int = []
        for _ in range(config.num_satt_layers):
            self.pointent_int.append(BaseSelfAttentionBlock(config))
        self.pointent_int = nn.Sequential(*self.pointent_int)

        self.pointnet_out = [
            nn.Linear(num_vectors * embed_size, num_vectors * (embed_size // 8), bias=True),
            nn.ReLU(),
            nn.Linear(num_vectors * (embed_size // 8), embed_size)
        ]
        self.pointnet_out = nn.Sequential(*self.pointnet_out)

    def forward(self, x, **kwargs):
        # B, N, P, D
        # B, N, P, D
        B, N, P, D = x.size()
        x = x.reshape(B * N, P, D)
        x = self.pointnet(x)

        # Vector interaction / aggregation
        x = x.reshape(B, N, P, -1)

        adj = kwargs.get('adj')
        if not adj is None:
            num_neighbors = adj.sum(dim=-1, keepdims=True).type(torch.float)
            x = adj @ x
            x /= num_neighbors
            x = torch.nan_to_num(x, nan=0.0)

        # Vector feature extraction
        x = self.pointent_int(x)

        # Full context output feature with pseudo-aggregation (ideally, here, we'd do a maxpool
        # over labels or regions and then pass them to the MLP)
        x = x.reshape(B, N, -1)
        x = self.pointnet_out(x)
        return x