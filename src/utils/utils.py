import cv2
import numpy as np
import imageio.v2 as imageio
import json 
import os
import pickle
import random
import torch
import warnings

import amelia_viz.common as C
import amelia_viz.marginal_predictions as M

from easydict import EasyDict
from importlib.util import find_spec
from omegaconf import DictConfig
from torch import tensor
from typing import Callable, Tuple
from geographiclib.geodesic import Geodesic

from src.utils import pylogger, rich_utils
from src.utils import global_masks as G
from src.utils.transform_utils import xy_to_ll

log = pylogger.get_pylogger(__name__)

def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:

        ...

        return metric_dict, object_dict
    ```
    """
    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap

def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value

def separate_ego_agent(value, ego_list):
    B = value.shape[0]
    batch_index   = range(B)
    ego_value = value[batch_index, ego_list] # B, T, N, D
    ego_value = ego_value.unsqueeze(1) # B, 1, T, N, D
    shape = ego_value.shape
    assert shape[0] == B and shape[1] == 1
    return ego_value

def plot_scene_batch(
    asset_dir: str, 
    batch: Tuple, 
    predictions: tensor, 
    hist_len: int, 
    geodesic: Geodesic, 
    tag: str, 
    out_dir: str = './out', 
    propagation: str = 'marginal', 
    dim: int = 2, 
    plot_full_scene = False,
    k_agents:int = 5,
    plot_n: int = 20
) -> None:
    scene = batch['scene_dict']
    pred_scores, mus, sigmas = predictions 
    B, N, H = pred_scores.shape

    # TODO: pre-load these in trajpred.py
    rasters, ref_ll, extent_ll = {}, {}, {}
    agents = {
        C.AIRCRAFT: imageio.imread(os.path.join(asset_dir, 'ac.png')),
        C.VEHICLE: imageio.imread(os.path.join(asset_dir, 'vc.png')),
        C.UNKNOWN: imageio.imread(os.path.join(asset_dir, 'uk_ac.png'))
    } 
    
    # TODO: update actual asset 
    image = agents[C.UNKNOWN]
    image = image.astype(np.float32)
    image = image * 1.35
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)    
    agents[C.UNKNOWN] = image
    
    if plot_full_scene:
        num_agents = scene['num_agents'][0]
        zipped = zip(
            pred_scores,           # B, N, H
            mus,                   # B, N, T, H, Dxy
            sigmas,                # B, N, T, H, Dxy
            scene['sequences'],    # B, N, T, D=9
            scene['num_agents'],   # B
            scene['ego_agent_id'], # B
            scene['agent_types'].numpy().reshape(B,num_agents),  # B
            scene['agents_in_scene'].numpy().astype(int).reshape(B,k_agents).tolist(),
            scene['airport_id'],   # B
            scene['scenario_id']   # B
        )
    
    else:
        zipped = zip(
            pred_scores,           # B, N, H
            mus,                   # B, N, T, H, Dxy
            sigmas,                # B, N, T, H, Dxy
            scene['sequences'],    # B, N, T, D=9
            scene['num_agents'],   # B
            scene['ego_agent_id'], # B
            scene['agent_types'],  # B
            scene['airport_id'],   # B
            scene['scenario_id']   # B
        )

    to_plot = random.sample(range(B), k=min(plot_n, B))
    
    for i, scene in enumerate(zipped):
        if plot_full_scene:
            (scores, mu, sigma, sequences, num_agents, ego_id, 
            agent_types, agents_in_scene ,airport, scenario_id) = scene
        
        else:
            (scores, mu, sigma, sequences, 
             num_agents, ego_id, agent_types, airport, scenario_id) = scene
            agents_in_scene = []
        
        if not i in to_plot:
            continue
        # TODO: preload these assets in trajpred.py
        if rasters.get(airport) is None:
            im = cv2.imread(os.path.join(asset_dir, airport, 'bkg_map.png'))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (im.shape[0]//2, im.shape[1]//2))
            rasters[airport] = im
        
            with open(os.path.join(asset_dir, airport,'limits.json'), 'r') as fp:
                ref_data = EasyDict(json.load(fp))
            ref_ll[airport] = [ref_data.ref_lat, ref_data.ref_lon, ref_data.range_scale]
            espg = ref_data.espg_4326
            extent_ll[airport] = (espg.north, espg.east, espg.south, espg.west)

        maps = (rasters[airport].copy(), agents)
        # Read sequences from batch
        gt_abs_traj = sequences[:num_agents] # N, T, D
        gt_history, gt_future = gt_abs_traj[:, :hist_len, :], gt_abs_traj[:, hist_len:, :]
        mu, sigma = mu[:num_agents, ..., :dim], sigma[:num_agents, ..., :dim]
        scores = scores[:num_agents]

        # Transform relative XY prediction to absolute LL space
        ll_pred, sigma_p, sigma_n = torch.zeros_like(mu), torch.zeros_like(mu), torch.zeros_like(mu)
        start_abs = gt_abs_traj[ego_id, hist_len-1, G.XY].detach().cpu().numpy()
        start_heading = gt_abs_traj[ego_id, hist_len-1, G.HD].detach().cpu().numpy()
        ref = ref_ll[airport]
        for h in range(H):
            ll_pred[:, :, h] = xy_to_ll(mu[:, :, h], start_abs, start_heading, ref, geodesic)
            
            mu_p = mu[:, :, h] + torch.sqrt(sigma[:, :, h])
            sigma_p[:, :, h] =  xy_to_ll(mu_p, start_abs, start_heading, ref, geodesic)

            mu_n = mu[:, :, h] - torch.sqrt(sigma[:, :, h])
            sigma_n[:, :, h] =  xy_to_ll(mu_n, start_abs, start_heading, ref, geodesic)
            
        sigma_np = torch.stack((sigma_n, sigma_p), dim=-1)
        tag_i = f"{airport}_scene-{i}_{scenario_id}_{tag}"
        
        if propagation == 'marginal':
            M.plot_scene_marginal_fast(
                gt_history = gt_history[..., G.HLL].detach().cpu().numpy(), 
                gt_future = gt_future[:, :, G.LL].detach().cpu().numpy(), 
                pred_trajectories = ll_pred[:, hist_len:].detach().cpu().numpy(), 
                pred_scores = scores.detach().cpu().numpy(), 
                sigmas = sigma_np[:, hist_len:].detach().cpu().numpy(), 
                maps = maps, ll_extent = extent_ll[airport], tag = tag_i, ego_id = ego_id, 
                out_dir = out_dir,
                agent_types=agent_types, 
                agents_interest= agents_in_scene
            )
        else:
            raise NotImplementedError(f"Propagation: {propagation}")
        

def load_assets(map_dir: str) -> Tuple:
    raster_map_filepath = os.path.join(map_dir, "bkg_map.png")
    raster_map = cv2.imread(raster_map_filepath)
    raster_map = cv2.resize(raster_map, (raster_map.shape[0]//2, raster_map.shape[1]//2))
    raster_map = cv2.cvtColor(raster_map, cv2.COLOR_BGR2RGB)

    pickle_map_filepath = os.path.join(map_dir, "semantic_graph.pkl")
    with open(pickle_map_filepath, 'rb') as f:
        graph_pickle = pickle.load(f)
        hold_lines = graph_pickle['hold_lines']
        graph_nx = graph_pickle['graph_networkx']
        # pickle_map = temp_dict['map_infos']['all_polylines'][:]
        
    limits_filepath = os.path.join(map_dir, 'limits.json')
    with open(limits_filepath, 'r') as fp:
        ref_data = EasyDict(json.load(fp))
    limits = (ref_data.north, ref_data.east, ref_data.south, ref_data.west)

    aircraft_filepath = os.path.join(map_dir, "ac.png")
    aircraft = imageio.imread(aircraft_filepath)

    vehicle_filepath = os.path.join(map_dir, "vc.png")
    vehicle = imageio.imread(vehicle_filepath)

    uk_filepath = os.path.join(map_dir, "uk.png")
    unknown =  imageio.imread(uk_filepath)
    
    agents = {AIRCRAFT: aircraft, VEHICLE: vehicle, UNKNOWN: unknown}
    return raster_map, hold_lines, graph_nx, limits, agents