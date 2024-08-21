import numpy as np
import os
import pandas as pd
import random
import torch

from datetime import datetime
from easydict import EasyDict
from math import floor
from typing import Tuple, List

from amelia_scenes.visualization.context import debug_plot
# from amelia_scenes.scene_utils.common import Status as s
from amelia_tf.utils.transform_utils import transform_points_2d

# TODO: figure out with the seeds from config are not working.
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None


def get_filtered_list(airport, base_dir, file_list, min_agents, max_agents):
    data_dirs = [os.path.join(base_dir, f) for f in file_list if airport in f]
    data_files = [os.path.join(dir_, f)
                  for dir_ in data_dirs for f in os.listdir(dir_)]
    # Each pickle file has a number at the end indicating the number of agents in the scenario.
    # Here, just return the files inside the min/max agent range.
    data_files = [f for f in data_files
                  if int(f.split('/')[-1].split('.')[0].split('-')[-1]) >= min_agents and
                  int(f.split('/')[-1].split('.')
                      [0].split('-')[-1]) <= max_agents
                  ]
    return data_files


def load_blacklist(data_prep: EasyDict, airport_list: list):
    """ Goes through the blacklist files and gets all blacklisted filenames for each airport. If no
        blacklists have been created, it'll only return an empty dictionary.

    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
        airport_list[list]: list of all supported airports in IATA code
    """
    # TODO: add blacklist path to configs/paths; this should handle automatic dir creation (I think)
    blacklist_dir = os.path.join(data_prep.in_data_dir, 'blacklist')
    os.makedirs(blacklist_dir, exist_ok=True)
    blacklist = {}
    for airport in airport_list:
        blackist_file = os.path.join(
            blacklist_dir, f"{airport}_{data_prep.split_type}.txt")
        blacklist[airport] = []
        if os.path.exists(blackist_file):
            with open(blackist_file, 'r') as f:
                blacklist[airport] = f.read().splitlines()
    return blacklist


def flatten_blacklist(blacklist: dict):
    blacklist_list = []
    for k, v in blacklist.items():
        blacklist_list += v
    return blacklist_list


def remove_blacklisted(blacklist: list, file_list: list):
    for duplicate in list(set(blacklist) & set(file_list)):
        file_list.remove(duplicate)
    return file_list


def get_airport_files(airport: str, data_prep: dict):
    """ Gets the airport data file list from the specified input directory and returns a random set.

    Inputs
    ------
        airport[str]: airport IATA
        data_prep[dict]: dictionary containing data preparation parameters.
    """
    in_data_dir = os.path.join(data_prep.in_data_dir, airport)
    airport_files = [os.path.join(airport, fp)
                     for fp in os.listdir(in_data_dir)]
    random.seed(data_prep.seed)
    random.shuffle(airport_files)
    return airport_files


def merge_seq3d_by_padding(
    tensor_list: List[torch.tensor], max_pad: int = None, return_pad_mask: bool = False
) -> torch.tensor:
    """ Merges a tensor list by padding whenever needed.

    Inputs
    ------
        tensor_list[torch.tensor]: list of tensors to be merged.
        max_pad[int]: if not None, sets the maximum padding.
        return_pad_mask[bool]: if True returns a mask of padded elements.

    Output
    ------
        ret_tensor[torch.tensor]: merged tensor.
        ret_mask[torch.tensor]: corresponding element mask.
    """
    assert len(tensor_list[0].shape) == 3
    max_num_agents = max([x.shape[0] for x in tensor_list]
                         ) if max_pad is None else max_pad

    num_agents, timesteps, dims = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == dims

        new_tensor = cur_tensor.new_zeros(max_num_agents, timesteps, dims)
        new_tensor[:cur_tensor.shape[0], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(max_num_agents)
        new_mask_tensor[:cur_tensor.shape[0]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.stack(ret_tensor_list, dim=0)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def merge_seq2d_by_padding(
    tensor_list: List[torch.tensor], max_pad: int = None, return_pad_mask: bool = False
) -> torch.tensor:
    """ Merges a tensor list by padding whenever needed.

    Inputs
    ------
        tensor_list[torch.tensor]: list of tensors to be merged.
        max_pad[int]: if not None, sets the maximum padding.
        return_pad_mask[bool]: if True returns a mask of padded elements.

    Output
    ------
        ret_tensor[torch.tensor]: merged tensor.
        ret_mask[torch.tensor]: corresponding element mask.
    """
    # TODO: debug this function
    assert len(tensor_list[0].shape) == 2
    max_num_agents = max([x.shape[0] for x in tensor_list]
                         ) if max_pad is None else max_pad

    num_agents, timesteps = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[-1] == timesteps

        new_tensor = cur_tensor.new_zeros(max_num_agents, timesteps)
        new_tensor[:cur_tensor.shape[0]] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(max_num_agents)
        new_mask_tensor[:cur_tensor.shape[0]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.stack(ret_tensor_list, dim=0)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def merge_seq1d_by_padding(
    tensor_list: List[torch.tensor], return_pad_mask: bool = False
) -> torch.tensor:
    """ Merges a tensor list by padding whenever needed.

    Inputs
    ------
        tensor_list[torch.tensor]: list of tensors to be merged.
        return_pad_mask[bool]: if True returns a mask of padded elements.

    Output
    ------
        ret_tensor[torch.tensor]: merged tensor.
        ret_mask[torch.tensor]: corresponding element mask.
    """
    assert len(tensor_list[0].shape) == 1
    max_size = max([x.shape[0] for x in tensor_list])

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]

        new_tensor = cur_tensor.new_zeros(max_size)
        new_tensor[:cur_tensor.shape[0]] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(max_size)
        new_mask_tensor[:cur_tensor.shape[0]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.cat(ret_tensor_list, dim=0).type(torch.float)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def extract_closest_points_from_query(
    points: np.array, query_point: Tuple, k_closest: int = 200
) -> np.array:
    """ Given a query point and a set of input points, returns the k closest ones.

    Inputs
    ------
        points[np.array(N, D)]: N input points.
        query_point[np.array(1, D)]: query point.
        k_closest[int]: number of closest points to return

    Output
    ------
        k_closest_points[np.array(k, D)]: extracted closest points.
    """
    dist = np.linalg.norm(points[:, 0:2] - query_point, axis=1)
    topk_idxs = dist.argsort()[:k_closest]
    return points[topk_idxs]


def compute_adjacency(context, adj_type='fully_connected', dist_threshold=0.050):
    P, D = context.shape
    if adj_type == 'fully_connected':
        return np.ones(shape=(P, P)).astype(float)
    elif adj_type == 'dist':
        from sklearn.metrics.pairwise import euclidean_distances
        c = context[:, :2]
        dist_matrix = euclidean_distances(c, c)
        adj = np.zeros(shape=(P, P))
        adj[np.where(dist_matrix <= dist_threshold)] = 1
        return adj
    else:
        raise NotImplementedError


def compute_local_context_from_ego_agent(
    global_context: np.array, ego_position: np.array, ego_heading: float, rel_sequences: np.array,
    current_time_step: int, num_local_points: int, debug: bool = False, **kwargs
) -> Tuple[np.array, list]:
    """ Generates a global context map transformed to the ego-agent frame.

    Inputs
    ------
        sequences[np.array]: array containing all agent's sequences.
        agent_id[int]: index of the ego agent in the sequence.
        heading[float]: heading of the ego agent.
        rel_sequences[np.array]: All polylines corresponding to the scenario dictionary.

    Outputs
    -------
        semantic_map[np.array]: the transformed map to ego-agent frame.
    """
    num_agents, _, _ = rel_sequences.shape

    # ego_agent's heading [rad] was already extracted and passed as input to
    # compute_scene_patches_from_ego_agent (by swim_dataset.py)
    tf_global_context = transform_points_2d(
        points=global_context, ref_point=ego_position, theta=ego_heading)

    local_context_list = []
    adjacency_list = []
    for n in range(num_agents):
        # Get the agent trajectory in Latitude/Longitude and then convert it to image coordinates
        x, y = rel_sequences[n, current_time_step, 0], rel_sequences[n, current_time_step, 1]
        local_context = extract_closest_points_from_query(
            points=tf_global_context, query_point=[x, y], k_closest=num_local_points)
        adjacency = compute_adjacency(local_context, adj_type='fully_connected')
        local_context_list.append(local_context)
        adjacency_list.append(adjacency)

    if debug:
        ego_id = kwargs.get('ego_id')
        limits = kwargs.get('limits')
        assert not ego_id is None and not limits is None
        debug_plot(ego_id, rel_sequences, tf_global_context, local_context_list, limits)

    return np.asarray(local_context_list), np.asarray(adjacency_list)

# def get_safety_score(sequences, agent_types, graph, airport, ref_data):
#     agent_types = np.array(agent_types)
#     hl_xy =  graph['hold_lines'][:, 2:4]
#     graph_map = graph['graph_networkx']
#     rwy_ext = np.max(ref_data.runway_extents)
#     # rwy_ext = C.RUNWAY_EXTENTS[airport]["max"]

#     #Compute individual metrics and scores
#     ind_metrics = indm.compute_individual_metrics(
#         sequences= sequences.copy(), hold_lines=hl_xy.copy())

#     ind_scores, ind_scene_score = S.compute_individual_scores(
#         metrics= ind_metrics, agent_types=agent_types)

#     # Compute interaction metrics and scores
#     int_metrics = intm.compute_interaction_metrics(
#         sequences= sequences.copy(), agent_types=agent_types, hold_lines=hl_xy.copy(),
#         graph_map=graph_map, agent_to_agent_dist_thresh=rwy_ext)

#     if np.where(int_metrics['status'] == s.OK)[0].shape[0] == 0:
#         return ind_scores, ind_scene_score

#     int_metrics['num_agents'] = agent_types.shape[0]
#     int_scores, int_scene_score = S.compute_interaction_scores(metrics=int_metrics)

#     # Compute individual and interaction scores
#     scores = ind_scores + int_scores
#     scene_score = ind_scene_score + int_scene_score
#     return scores, scene_score
