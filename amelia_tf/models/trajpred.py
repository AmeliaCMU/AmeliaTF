import itertools
import numpy as np
import os
import torch
import torch.nn as nn

from datetime import date
from easydict import EasyDict
from geographiclib.geodesic import Geodesic
from lightning import LightningModule
from torchmetrics import MeanMetric
from typing import Any

from amelia_tf.models.components.common import LayerNorm
from amelia_tf.utils.utils import plot_scene_batch
from amelia_tf.utils import global_masks as G
from amelia_tf.utils.utils import separate_ego_agent

np.printoptions(precision=5, suppress=True)


class TrajPred(LightningModule):
    """ Trajectory Prediction module wrapper based on:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
        net: torch.nn.Module, extra_params: EasyDict
    ):
        """ Initializes the trajectory prediction module.

        Inputs
        ------
            optimizer[torch.optim.Optimizer]: optimizer object.
            scheduler[torch.optim.lr_scheduler]: learning rate scheduler.
            net[torch.nn.Module]: model object.
            extra_params[EasyDict]: dictionary containing all other parameters needed by the module.
        """
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute also ensures init
        # params will be stored in ckpt
        self.save_hyperparameters(ignore=['net'], logger=False)

        self.net = net
        self.hist_len = self.net.hist_len
        self.pred_lens = self.net.pred_lens
        self.num_dec_heads = self.net.num_dec_heads

        self.eparams = extra_params
        self.seen_airports = self.eparams.seen_airports
        self.unseen_airports = self.eparams.unseen_airports

        # For averaging loss across batches
        self.train_loss, self.val_loss, self.test_loss = MeanMetric(), MeanMetric(), MeanMetric()

        # For tracking best so far validation and testing accuracy
        self.max_pred_len = max(self.pred_lens)
        self.val_ade, self.test_ade, self.val_fde, self.test_fde = {}, {}, {}, {}

        # collision
        # gt: ground truth
        self.val_coll_pred2gt = {}
        self.test_coll_pred2gt05, self.test_coll_pred2gt1, self.test_coll_pred2gt3 = {}, {}, {}
        self.test_coll_pred2pred05, self.test_coll_pred2pred1, self.test_coll_pred2pred3 = {}, {}, {}
        self.test_coll_pred2gt2gt05, self.test_coll_pred2gt2gt1, self.test_coll_pred2gt2gt3 = {}, {}, {}

        for t in self.pred_lens:
            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.val_ade[key], self.test_ade[key] = MeanMetric(), MeanMetric()
            self.val_fde[key], self.test_fde[key] = MeanMetric(), MeanMetric()

            self.val_coll_pred2gt[key], self.test_coll_pred2gt05[key] = MeanMetric(), MeanMetric()
            self.test_coll_pred2gt1[key], self.test_coll_pred2gt3[key] = MeanMetric(), MeanMetric()
            self.test_coll_pred2pred05[key], self.test_coll_pred2pred1[key] = MeanMetric(), MeanMetric()
            self.test_coll_pred2pred3[key] = MeanMetric()
            self.test_coll_pred2gt2gt05[key], self.test_coll_pred2gt2gt1[key] = MeanMetric(), MeanMetric()
            self.test_coll_pred2gt2gt3[key] = MeanMetric()
        self.val_ade, self.test_ade = nn.ModuleDict(self.val_ade), nn.ModuleDict(self.test_ade)
        self.val_fde, self.test_fde = nn.ModuleDict(self.val_fde), nn.ModuleDict(self.test_fde)

        self.val_coll_pred2gt = nn.ModuleDict(self.val_coll_pred2gt)
        self.test_coll_pred2gt05 = nn.ModuleDict(self.test_coll_pred2gt05)
        self.test_coll_pred2pred05 = nn.ModuleDict(self.test_coll_pred2pred05)
        self.test_coll_pred2gt1 = nn.ModuleDict(self.test_coll_pred2gt1)
        self.test_coll_pred2pred1 = nn.ModuleDict(self.test_coll_pred2pred1)
        self.test_coll_pred2gt3 = nn.ModuleDict(self.test_coll_pred2gt3)
        self.test_coll_pred2pred3 = nn.ModuleDict(self.test_coll_pred2pred3)
        self.test_coll_pred2gt2gt05 = nn.ModuleDict(self.test_coll_pred2gt2gt05)
        self.test_coll_pred2gt2gt1 = nn.ModuleDict(self.test_coll_pred2gt2gt1)
        self.test_coll_pred2gt2gt3 = nn.ModuleDict(self.test_coll_pred2gt2gt3)

        # self.val_prob_ade, self.test_prob_ade = MeanMetric(), MeanMetric()
        # self.val_prob_fde, self.test_prob_fde = MeanMetric(), MeanMetric()

        self.val_seen_ade, self.test_seen_ade = {}, {}
        self.val_seen_fde, self.test_seen_fde = {}, {}
        for pred_len, airport in itertools.product(self.pred_lens, self.seen_airports):
            key = f"{airport}_t={pred_len}"
            self.val_seen_ade[key], self.test_seen_ade[key] = MeanMetric(), MeanMetric()
            self.val_seen_fde[key], self.test_seen_fde[key] = MeanMetric(), MeanMetric()
        self.val_seen_ade = nn.ModuleDict(self.val_seen_ade)
        self.val_seen_fde = nn.ModuleDict(self.val_seen_fde)
        self.test_seen_ade = nn.ModuleDict(self.test_seen_ade)
        self.test_seen_fde = nn.ModuleDict(self.test_seen_fde)

        # Create metrics for unseen airports
        if len(self.unseen_airports) > 0:
            self.test_unseen_ade, self.test_unseen_fde = {}, {}
            for pred_len, airport in itertools.product(self.pred_lens, self.unseen_airports):
                key = f"{airport}_t={pred_len}"
                self.test_unseen_ade[key], self.test_unseen_fde[key] = MeanMetric(), MeanMetric()
            self.test_unseen_ade = nn.ModuleDict(self.test_unseen_ade)
            self.test_unseen_fde = nn.ModuleDict(self.test_unseen_fde)

        assert self.eparams.propagation in ['joint', 'marginal']
        if self.eparams.propagation == 'marginal':
            from amelia_tf.utils.metrics import marginal_ade as ade
            from amelia_tf.utils.metrics import marginal_fde as fde
            from amelia_tf.utils.metrics import marginal_prob_ade as prob_ade
            from amelia_tf.utils.metrics import marginal_prob_fde as prob_fde
            from amelia_tf.utils.losses import marginal_loss as compute_loss
        else:
            from amelia_tf.utils.metrics import joint_ade as ade
            from amelia_tf.utils.metrics import joint_fde as fde
            from amelia_tf.utils.metrics import joint_prob_ade as prob_ade
            from amelia_tf.utils.metrics import joint_prob_fde as prob_fde
            from amelia_tf.utils.losses import lmbd_marginal_joint_loss as compute_loss

        self.ade, self.fde, self.prob_ade, self.prob_fde = ade, fde, prob_ade, prob_fde
        self.compute_loss = compute_loss
        self.geodesic = Geodesic.WGS84

        os.makedirs(self.eparams.plot_dir, exist_ok=True)
        out_dir = os.path.join(self.eparams.plot_dir, f"{date.today()}_{self.eparams.tag}")
        self.val_out_dir = os.path.join(out_dir, 'val')
        os.makedirs(self.val_out_dir, exist_ok=True)
        self.test_out_dir = os.path.join(out_dir, 'test')
        os.makedirs(self.test_out_dir, exist_ok=True)

    def on_train_start(self):
        """ by default lightning executes validation step sanity checks before training starts, so
        it's worth to make sure validation metrics don't store results from these checks. """
        self.val_loss.reset()

    def model_step(self, batch, plot: bool = False, tag: str = 'temp', out_dir: str = 'temp'):
        """ Runs the model's forward function and then computes the loss function. If plot is True
        it will run and save scene visualizations.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            plot[bool]: if True, it visualizes the scene.
            out_dir[str]: output directory.
            tag[str]: tag name to save the output file.

        Output
        ------
            loss[torch.tensor]: model's loss value.
            pred_scores[torch.tensor]: predictions scores.
            mu[torch.tensor]: predicted means.
            sigma[torch.tensor]: predicted standard deviations.
            Y_out[torch.tensor]: ground truth futures.
        """
        # TODO: roll up ego-agent. TF not viewpoint invariant.
        # (B, N, T, D)
        Y = batch['scene_dict']['rel_sequences']
        X = torch.zeros_like(Y).type(torch.float)
        X[:, :, :self.hist_len] = Y[:, :, :self.hist_len]

        # -----------------------------------------
        # TODO: incorporate heading prediction
        B, N, T, D = Y.shape
        Y = Y[..., G.REL_XYZ[:D]]
        context = batch['scene_dict']['context']
        adjacency = batch['scene_dict']['adjacency']
        ego_agent = batch['scene_dict']['ego_agent_id']
        masks = batch['scene_dict']['agent_masks']

        # TODO: address attention-based masking
        pred_scores, mu, sigma = self.net(
            X, context=context, adjacency=adjacency,
            mask=None,  # batch['scene_dict']['agent_masks'] if self.eparams.use_agent_masks else None
        )
        loss = self.compute_loss(
            pred_scores, mu, sigma, Y, ego_agent=ego_agent, epoch=self.current_epoch+1,
            agent_mask=batch['scene_dict']['agent_masks'],
        )

        if plot:
            predictions = (pred_scores, mu, sigma)
            plot_scene_batch(
                self.eparams.asset_dir, batch, predictions, self.hist_len, self.geodesic, tag,
                out_dir, self.eparams.propagation
            )

        return loss, pred_scores, mu, sigma, Y[:, :, self.hist_len:]

    def training_step(self, batch: Any, batch_idx: int):
        """ Performs a model step on a training batch.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            batch_idx[int]: index of current batch.

        Output
        ------
            loss[torch.tensor]: model's loss value.
        """
        loss, _, _, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("losses/train", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        """ Performs a model step on a validation batch.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            batch_idx[int]: index of current batch.
        """
        plot = self.eparams.plot_val \
            if self.current_epoch >= self.eparams.plot_after_n_epochs \
            and (batch_idx+1) % self.eparams.plot_every_n == 0 else False

        tag = f"epoch-{self.current_epoch}_batch-idx{batch_idx}"
        loss, pred_scores, mu, sigma, fut_rel = self.model_step(batch, plot, tag, self.val_out_dir)

        # Separate ego agent prediction
        if self.eparams.propagation == 'marginal':
            ego_agent = batch['scene_dict']['ego_agent_id']
            ego_mu = separate_ego_agent(mu, ego_agent)
            ego_sigma = separate_ego_agent(sigma, ego_agent)
            ego_pred_scores = separate_ego_agent(pred_scores, ego_agent)
            ego_fut = separate_ego_agent(fut_rel, ego_agent)
            mask = separate_ego_agent(batch['scene_dict']['agent_masks'], ego_agent)
        else:
            raise NotImplementedError

        self.val_loss(loss)
        self.log("losses/val", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for t in self.pred_lens:
            mu_t = ego_mu[:, :, :self.hist_len+t]
            mask_t = mask[:, :, :self.hist_len+t]
            fut_t = ego_fut[:, :, :t]

            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.val_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.log(f"val_ade/{key}", self.val_ade[key], on_step=False, on_epoch=True, prog_bar=True)

            self.val_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(f"val_fde/{key}", self.val_fde[key], on_step=False, on_epoch=True, prog_bar=True)

        # self.val_prob_ade(self.prob_ade(ego_mu, ego_pred_scores, ego_fut, mask=mask))
        # self.log("val/prob_ade", self.val_prob_ade, on_step=False, on_epoch=True, prog_bar=True)

        # self.val_prob_fde(self.prob_fde(ego_mu, ego_pred_scores, ego_fut, mask=mask))
        # self.log("val/prob_fde", self.val_prob_fde, on_step=False, on_epoch=True, prog_bar=True)

        if len(self.seen_airports) > 1:
            airport_ids = batch['scene_dict']['airport_id']
            for airport in self.seen_airports:
                airport_idx = np.where(airport_ids == airport)[0]
                if len(airport_idx) == 0:
                    continue
                airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
                airport_mask = mask[airport_idx]

                for t in self.pred_lens:
                    mu_t = airport_mu[:, :, :self.hist_len+t]
                    fut_t = airport_fut[:, :, :t]
                    mask_t = airport_mask[:, :, :self.hist_len+t]

                    key = f"{airport}_t={t}"
                    self.val_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"val_seen_ade/{key}", self.val_seen_ade[key], on_step=False, on_epoch=True,
                        prog_bar=True)

                    self.val_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"val_seen_fde/{key}", self.val_seen_fde[key], on_step=False, on_epoch=True,
                        prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """ Performs a model step on a test batch.

        Inputs
        ------
            batch[Any]: dictionary containing the batch parameters.
            batch_idx[int]: index of current batch.
        """
        plot = self.eparams.plot_test if (batch_idx+1) % self.eparams.plot_every_n == 0 else False

        tag = f"epoch-{self.current_epoch}_batch-idx{batch_idx}"
        loss, pred_scores, mu, sigma, fut_rel = self.model_step(batch, plot, tag, self.test_out_dir)
        ego_agent = batch['scene_dict']['ego_agent_id']

        if self.eparams.propagation == 'marginal':
            # Separate ego agent prediction
            num_agents = batch['scene_dict']['num_agents']
            ego_mu = separate_ego_agent(mu, ego_agent)
            ego_sigma = separate_ego_agent(sigma, ego_agent)
            ego_pred_scores = separate_ego_agent(pred_scores, ego_agent)
            ego_fut = separate_ego_agent(fut_rel, ego_agent)
            mask = None if not 'agent_masks' in batch['scene_dict'].keys() else \
                separate_ego_agent(batch['scene_dict']['agent_masks'], ego_agent)
        else:
            raise NotImplementedError

        self.test_loss(loss)
        self.log("losses/test", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

        for t in self.pred_lens:
            mu_t, fut_t = ego_mu[:, :, :self.hist_len+t], ego_fut[:, :, :t]
            mask_t = None if mask is None else mask[:, :, :self.hist_len+t]

            mus_t = mu[:, :, :self.hist_len+t]

            key = 't=max' if t == self.max_pred_len else f"t={t}"
            self.test_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
            self.log(f"test_ade/{t}", self.test_ade[key], on_step=False, on_epoch=True, prog_bar=True)

            self.test_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
            self.log(f"test_fde/{t}", self.test_fde[key], on_step=False, on_epoch=True, prog_bar=True)

            self.test_coll_pred2gt05[key](
                self.coll_pred2gt(mu_t, fut_rel[:, :, :t], num_agents, ego_agent, coll_thresh=0.05))
            self.log(
                f"test_coll_pred2gt0.05/{key}", self.test_coll_pred2gt05[key], on_step=False, on_epoch=True,
                prog_bar=True)

            self.test_coll_pred2pred05[key](
                self.coll_pred2pred(mus_t, fut_rel[:, :, :t], num_agents, ego_agent, coll_thresh=0.05))
            self.log(
                f"test_coll_pred2pred0.05/{key}", self.test_coll_pred2pred05[key], on_step=False, on_epoch=True,
                prog_bar=True)

            self.test_coll_pred2gt2gt05[key](
                self.coll_pred2gt2gt(fut_t, fut_rel[:, :, :t], num_agents, ego_agent, coll_thresh=0.05))
            self.log(
                f"test_coll_pred2gt2gt0.05/{key}", self.test_coll_pred2gt2gt05[key], on_step=False, on_epoch=True,
                prog_bar=True)

            self.test_coll_pred2gt1[key](
                self.coll_pred2gt(mu_t, fut_rel[:, :, :t], num_agents, ego_agent, coll_thresh=0.1))
            self.log(
                f"test_coll_pred2gt0.1/{key}", self.test_coll_pred2gt1[key], on_step=False, on_epoch=True,
                prog_bar=True)

            self.test_coll_pred2pred1[key](
                self.coll_pred2pred(mus_t, fut_rel[:, :, :t], num_agents, ego_agent, coll_thresh=0.1))
            self.log(
                f"test_coll_pred2pred0.1/{key}", self.test_coll_pred2pred1[key], on_step=False, on_epoch=True,
                prog_bar=True)

            self.test_coll_pred2gt2gt1[key](
                self.coll_pred2gt2gt(fut_t, fut_rel[:, :, :t], num_agents, ego_agent, coll_thresh=0.1))
            self.log(
                f"test_coll_pred2gt2gt0.1/{key}", self.test_coll_pred2gt2gt1[key], on_step=False, on_epoch=True,
                prog_bar=True)

            self.test_coll_pred2gt3[key](
                self.coll_pred2gt(mu_t, fut_rel[:, :, :t], num_agents, ego_agent, coll_thresh=0.3))
            self.log(
                f"test_coll_pred2gt0.3/{key}", self.test_coll_pred2gt3[key], on_step=False, on_epoch=True,
                prog_bar=True)

            self.test_coll_pred2pred3[key](
                self.coll_pred2pred(mus_t, fut_rel[:, :, :t], num_agents, ego_agent, coll_thresh=0.3))
            self.log(
                f"test_coll_pred2pred0.3/{key}", self.test_coll_pred2pred3[key], on_step=False, on_epoch=True,
                prog_bar=True)

            self.test_coll_pred2gt2gt3[key](
                self.coll_pred2gt2gt(fut_t, fut_rel[:, :, :t], num_agents, ego_agent, coll_thresh=0.1))
            self.log(
                f"test_coll_pred2gt2gt0.3/{key}", self.test_coll_pred2gt2gt3[key], on_step=False, on_epoch=True,
                prog_bar=True)

        # self.test_prob_ade(self.prob_ade(ego_mu, ego_pred_scores, ego_fut))
        # self.log("test/prob_ade", self.test_prob_ade, on_step=False, on_epoch=True, prog_bar=True)

        # self.test_prob_fde(self.prob_fde(ego_mu, ego_pred_scores, ego_fut))
        # self.log("test/prob_fde", self.test_prob_fde, on_step=False, on_epoch=True, prog_bar=True)

        airport_ids = batch['scene_dict']['airport_id']
        for airport in self.seen_airports:
            airport_idx = np.where(airport_ids == airport)[0]
            if len(airport_idx) == 0:
                continue
            airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
            airport_mask = mask[airport_idx]

            for t in self.pred_lens:
                mu_t, fut_t = airport_mu[:, :, :self.hist_len+t], airport_fut[:, :, :t]
                mask_t = airport_mask[:, :, :self.hist_len+t]

                key = f"{airport}_t={t}"
                self.val_seen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                self.log(
                    f"test_seen_ade/{key}", self.val_seen_ade[key], on_step=False, on_epoch=True,
                    prog_bar=True)

                self.val_seen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                self.log(
                    f"test_seen_fde/{key}", self.val_seen_fde[key], on_step=False, on_epoch=True,
                    prog_bar=True)

        if len(self.unseen_airports) > 0:
            for airport in self.unseen_airports:
                airport_idx = np.where(airport_ids == airport)[0]
                if len(airport_idx) == 0:
                    continue
                airport_mu, airport_fut = ego_mu[airport_idx], ego_fut[airport_idx]
                airport_mask = mask[airport_idx]

                for t in self.pred_lens:
                    mu_t, fut_t = airport_mu[:, :, :self.hist_len+t], airport_fut[:, :, :t]
                    mask_t = airport_mask[:, :, :self.hist_len+t]

                    key = f"{airport}_t={t}"
                    self.test_unseen_ade[key](self.ade(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"test_unseen_ade/{key}", self.test_unseen_ade[key], on_step=False,
                        on_epoch=True, prog_bar=True)

                    self.test_unseen_fde[key](self.fde(mu_t, fut_t, mask=mask_t))
                    self.log(
                        f"test_unseen_fde/{key}", self.test_unseen_fde[key], on_step=False,
                        on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """ This long function is unfortunately doing something very simple and is being very
        defensive: We are separating out all parameters of the model into two buckets: those that
        will experience weight decay for regularization and those that won't (biases, layernorm,
        embedding weights). We are then returning the PyTorch optimizer object.

        NOTE: For reference as to why this function is needed:
            https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025
            https://discuss.pytorch.org/t/ \
                weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
        """
        # separate out all parameters that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.Conv1d)
        blacklist_weight_modules = (
            torch.nn.SyncBatchNorm, nn.LayerNorm, LayerNorm, nn.Embedding, nn.BatchNorm1d,
            nn.BatchNorm2d, nn.MultiheadAttention
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.hparams.optimizer.weight_decay},
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.optimizer.lr,
            betas=(self.hparams.optimizer.beta1, self.hparams.optimizer.beta2)
        )

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "losses/val",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {
            "optimizer": optimizer
        }


<< << << < HEAD: src/models/trajpred.py

== == == =
>>>>>> > src_main: amelia_tf/models/trajpred.py
if __name__ == "__main__":
    _ = TrajPred(None, None, None)
