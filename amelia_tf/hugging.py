from amelia_scenes.utils.dataset import load_assets
from amelia_scenes.visualization import scene_viz as viz
from amelia_scenes.visualization import common as cviz
from amelia_inference.utils.common import *
from amelia_inference.utils.data_utils import get_scene_list, get_full_scene_batch
from geographiclib.geodesic import Geodesic
from amelia_scenes.visualization.scene_viz import plot_scene
from amelia_tf.models.components.amelia_traj import AmeliaTraj  # Non context aware model
from amelia_tf.models.components.amelia import AmeliaTF  # Context aware model
from amelia_tf.data.components.amelia_dataset import AmeliaDataset
from amelia_tf.utils.utils import plot_scene_batch
import os
import torch
import random
import numpy as np
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

from amelia_inference.standalone import SocialTrajPred


# Load model and checkpoint


def get_inference_time(sequences, model, test_config, frame_limits):
    start_frame, end_frame = frame_limits
    print('---- Starting GPU Warmup ----')
    for i in range(0, 10):
        test_seq = sequences[i]
        if test_seq.size != 0:
            pred_scores, mu, sigma = model.forward(test_seq, False, f'test_{i}')
    timings = np.zeros(((end_frame - start_frame), 1))

    # Forward the remaining frames
    for i in tqdm(range(start_frame, end_frame)):
        test_seq = sequences[i]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if test_seq.size != 0:
            pred_scores, mu, sigma = model.forward(test_seq, test_config['plot'],
                                                   f'test_{i}', test_config['save_scene'])
        end.record()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        curr_time = start.elapsed_time(end)
        timings[i] = curr_time
    mean_syn = np.mean(timings)
    std_syn = np.std(timings)
    print('Mean: ', mean_syn)
    print('Std: ', std_syn)


def set_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(version_base="1.2", config_path="configs", config_name="default.yaml")
def main(cfg: DictConfig) -> None:
    dataloader = AmeliaDataset(cfg.data.config)
    set_seeds(cfg.seed)

    if cfg.tests.use_map:
        model = AmeliaTF(cfg.model.config)
    else:
        model = AmeliaTraj(cfg.model.config)

    # Load extra configuration parameters
    from_pickle = cfg.tests.from_pickle
    plot = cfg.tests.plot

    device = torch.device(cfg.device.accelerator)
    # Load models
    predictor = SocialTrajPred(cfg.tests.airport, model=model, dataloader=dataloader,
                               use_map=cfg.tests.use_map, device=device)
    predictor.load_ckpt(cfg.tests.ckpt_path, from_pickle)

    print(f"----- Loading scenes for {cfg.tests.scene_file} -----")
    vis_tag = f"{cfg.tests.scene_file}_{cfg.tests.tag}"
    out_dir = os.path.join(cfg.tests.out_data_dir, f"{cfg.tests.scene_file}_{cfg.tests.tag}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    scene_file_list, scenes = get_scene_list(cfg.tests.scene_file, cfg.tests.airport,
                                             traj_dir=cfg.paths.proc_scenes,
                                             max_scenes=cfg.tests.max_scenes,
                                             sorted=False)
    print("----- Forwarding scenes -----")

    for scene in tqdm(scenes):
        batch, predictions = predictor.forward(
            scene, random_ego=cfg.tests.random_ego, ego_agent_id=cfg.tests.ego_agent_id)
        if plot:
            full_scene, preds = get_full_scene_batch(batch, scene, predictions, device)
            plot_scene_batch(
                asset_dir=cfg.data.config.assets_dir,
                batch=full_scene,
                predictions=preds,
                hist_len=model.hist_len,
                geodesic=Geodesic.WGS84,
                tag=vis_tag,
                plot_full_scene=True,
                out_dir=out_dir
            )


if __name__ == '__main__':
    main()
