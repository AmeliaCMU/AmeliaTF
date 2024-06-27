import numpy as np
import os
import pandas as pd 
import random 
import torch

from datetime import datetime
from easydict import EasyDict
from math import floor
from typing import Tuple, List

# import amelia_scenes.scene_utils.individual_metrics as indm
# import amelia_scenes.scene_utils.interaction_metrics as intm
# import amelia_scenes.scene_utils.scores as S

from amelia_viz.context import debug_plot
# from amelia_scenes.scene_utils.common import Status as s
from src.utils.transform_utils import transform_points_2d

# TODO: figure out with the seeds from config are not working. 
np.set_printoptions(suppress=True)
pd.options.mode.chained_assignment = None

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
        blackist_file = os.path.join(blacklist_dir, f"{airport}_{data_prep.split_type}.txt")
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
    airport_files = [os.path.join(airport, fp) for fp in os.listdir(in_data_dir)]
    random.seed(data_prep.seed)
    random.shuffle(airport_files)
    return airport_files

def create_random_splits(data_prep: EasyDict, airport_list: list):
    """ Splits the data by month. If no `test_airports` are specified, then it will iterate over all 
    `train_airports` and create a train-val-test split for each, by keeping floor(75%) of the files 
    into the train-val and the remaining floor(25%) into the test set. Files are randomly selected.
    
    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
        airport_list[list]: list of all supported airports in IATA code
    """
    n_train, n_val, n_test = data_prep.random_splits.train_val_test
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_prep.in_data_dir, f"{split}_splits")
        os.makedirs(split_dir, exist_ok=True)

    for airport in airport_list:
        airport_files = get_airport_files(airport, data_prep)

        N_train = floor(len(airport_files) * n_train)
        N_val = floor(len(airport_files) * n_val)

        filename = f"{airport}_{data_prep.split_type}"

        # Write the out the splits 
        train_list = airport_files[:N_train]
        with open(f"{data_prep.in_data_dir}/train_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(train_list))
        
        val_list = airport_files[N_train:N_train+N_val]
        with open(f"{data_prep.in_data_dir}/val_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(val_list))

        test_list = airport_files[N_train+N_val:]
        with open(f"{data_prep.in_data_dir}/test_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(test_list))


def create_day_splits(data_prep: dict, airport_list: list):
    """ Splits the data by month. If no `test_airports` are specified, then it will iterate over all 
    `seen_airports` and create a train-val-test split for each, by keeping floor(75%) of the days 
    into the train-val and the remaining floor(25%) into the test set. 
    
    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
        airport_list[list]: list of all supported airports in IATA code
    """
    n_train, n_val = data_prep.day_splits.train_val
    train_val_perc = data_prep.day_splits.train_val_perc
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_prep.in_data_dir, f"{split}_splits")
        os.makedirs(split_dir, exist_ok=True)

    for airport in airport_list:
        # Collect all airport files in current airport and get the unique days for which data was 
        # collected. 
        airport_files = np.asarray(get_airport_files(airport, data_prep))
        days_per_file = np.asarray([datetime.utcfromtimestamp(
            int(f.split('/')[-1].split('.')[0].split('_')[-1])).day for f in airport_files])
        days = np.unique(days_per_file)
        num_days = days.shape[0]

        np.random.seed(data_prep.seed)

        # Make sure test set does not contain days "seen" during training. 
        train_val_days = np.random.choice(days, size=int(train_val_perc * num_days), replace=False)
        test_days = list(set(days.tolist()).symmetric_difference(train_val_days.tolist()))

        train_val_idx = np.in1d(days_per_file, train_val_days)
        train_val_files = airport_files[train_val_idx].tolist()
        
        filename = f"{airport}_{data_prep.split_type}"

        N_train = floor(len(train_val_files) * n_train)
        train_list = train_val_files[:N_train]
        with open(f"{data_prep.in_data_dir}/train_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(train_list))

        val_list = train_val_files[N_train:]
        with open(f"{data_prep.in_data_dir}/val_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(val_list))

        test_idx = np.in1d(days_per_file, test_days)
        test_list = airport_files[test_idx].tolist()
        with open(f"{data_prep.in_data_dir}/test_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(test_list))

def create_month_splits(data_prep: dict, airport_list: list):
    """ Splits the data by month. If no `test_airports` are specified, then it will iterate over all 
    `seen_airports` and create a train-val-test split for each, by keeping floor(75%) of the days 
    into the train-val and the remaining floor(25%) into the test set. 
    
    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
        airport_list[list]: list of all supported airports in IATA code
    """
    n_train, n_val = data_prep.day_splits.train_val
    train_val_perc = data_prep.day_splits.train_val_perc
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_prep.in_data_dir, f"{split}_splits")
        os.makedirs(split_dir, exist_ok=True)

    for airport in airport_list:
        # Collect all airport files in current airport and get the unique days for which data was 
        # collected. 
        airport_files = np.asarray(get_airport_files(airport, data_prep))
        month_per_file = np.asarray([datetime.utcfromtimestamp(
            int(f.split('/')[-1].split('.')[0].split('_')[-1])).month for f in airport_files])
        months = np.unique(month_per_file)
        num_months = months.shape[0]
        
        np.random.seed(data_prep.seed)

        # Make sure test set does not contain months "seen" during training. 
        train_val_months = np.random.choice(
            months, size=int(train_val_perc * num_months), replace=False)
        test_months = list(set(months.tolist()).symmetric_difference(train_val_months.tolist()))

        train_val_idx = np.in1d(month_per_file, train_val_months)
        train_val_files = airport_files[train_val_idx].tolist()
        
        filename = f"{airport}_{data_prep.split_type}"

        N_train = floor(len(train_val_files) * n_train)
        train_list = train_val_files[:N_train]
        with open(f"{data_prep.in_data_dir}/train_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(train_list))

        val_list = train_val_files[N_train:]
        with open(f"{data_prep.in_data_dir}/val_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(val_list))

        test_idx = np.in1d(month_per_file, test_months)
        test_list = airport_files[test_idx].tolist()
        with open(f"{data_prep.in_data_dir}/test_splits/{filename}.txt", 'w') as fp:
            fp.write('\n'.join(test_list))

def split_data_by_day(data_prep: dict):
    """ Splits the data by month. If no `test_airports` are specified, then it will iterate over all 
    `seen_airports` and create a train-val-test split for each, by keeping floor(75%) of the days 
    into the train-val and the remaining floor(25%) into the test set. 
    
    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
    """
    train_list, val_list, test_list = [], [], []
    n_train, n_val = data_prep.day_splits.train_val

    for airport in data_prep.seen_airports:
        # Collect all airport files in current airport and get the unique days for which data was 
        # collected. 
        airport_files = np.asarray(get_airport_files(airport, data_prep))
        days_per_file = np.asarray([datetime.utcfromtimestamp(
            int(f.split('/')[-1].split('.')[0].split('_')[-1])).day for f in airport_files])
        days = np.unique(days_per_file)
        num_days = days.shape[0]
        
        # TODO: seed globally. 
        np.random.seed(data_prep.seed)

        # Make sure test set does not contain days "seen" during training. 
        train_val_days = np.random.choice(
            days, size=int(data_prep.day_splits.train_val_perc * num_days), replace=False)
        test_days = list(set(days.tolist()).symmetric_difference(train_val_days.tolist()))

        train_val_idx = np.in1d(days_per_file, train_val_days)
        train_val_files = airport_files[train_val_idx].tolist()
        
        N_train = floor(len(train_val_files) * n_train)
        train_list += train_val_files[:N_train]
        val_list += train_val_files[N_train:]

        test_idx = np.in1d(days_per_file, test_days)
        test_list += airport_files[test_idx].tolist()
    return train_list, val_list, test_list

def split_data_by_month(data_prep: dict):
    """ Splits the data by month. If no `test_airports` are specified, then it will iterate over all 
    `train_airports` and create a train-val-test split for each, by keeping floor(75%) of the months 
    into the train-val and the remaining floor(25%) into the test set. 
    
    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
    """
    train_list, val_list, test_list = [], [], []
    n_train, n_val = data_prep.month_splits.train_val

    for airport in data_prep.train_airports:
        # select the months
        airport_files = np.asarray(get_airport_files(airport, data_prep))
        month_per_file = np.asarray([datetime.utcfromtimestamp(
            int(f.split('/')[-1].split('.')[0].split('_')[-1])).month for f in airport_files])
        months = np.unique(month_per_file)
        num_months = months.shape[0]
        
        train_val_months = np.random.choice(
            months, size=int(data_prep.month_splits.train_val_perc * num_months), replace=False)
        test_months = list(set(months.tolist()).symmetric_difference(train_val_months.tolist()))

        train_val_idx = np.in1d(month_per_file, train_val_months)
        train_val_files = airport_files[train_val_idx].tolist()
        
        N_train = floor(len(train_val_files) * n_train)
        train_list += train_val_files[:N_train]
        val_list += train_val_files[N_train:]

        test_idx = np.in1d(month_per_file, test_months)
        test_list += airport_files[test_idx].tolist()
    return train_list, val_list, test_list

def split_data_randomly(data_prep):
    """ Splits the data by month. If no `test_airports` are specified, then it will iterate over all 
    `train_airports` and create a train-val-test split for each, by keeping floor(75%) of the files 
    into the train-val and the remaining floor(25%) into the test set. Files are randomly selected.
    
    Inputs
    ------
        data_prep[dict]: dictionary containing data preparation parameters.
    """
    train_list, val_list, test_list = [], [], []
    n_train, n_val = data_prep.random_splits.train_val

    for airport in data_prep.train_airports:
        airport_files = get_airport_files(airport)

        N_train = floor(len(airport_files) * n_train)
        N_val = floor(len(airport_files) * n_val)

        train_list += airport_files[:N_train]
        val_list += airport_files[N_train:N_train+N_val]
        test_list += airport_files[N_train+N_val:]
    return train_list, val_list, test_list

def impute(seq: pd.DataFrame, seq_len: int, imputed_flag: float = 1.0) -> pd.DataFrame:
    """ Imputes missing data via linear interpolation. 
    
    Inputs
    ------
        seq[pd.DataFrame]: trajectory sequence to be imputed.
        seq_len[int]: length of the trajectory sequence.
    
    Output
    ------
        seq[pd.DataFrame]: trajectory sequence after imputation.
    """
    # Create a list from starting frame to ending frame in agent sequence
    conseq_frames = set(range(int(seq[0, 0]), int(seq[-1, 0])+1))
    # Create a list of the actual frames in the agent sequence. There may be missing data from which
    # we need to interpolate.
    actual_frames = set(seq[:, 0])
    # Compute the difference between the lists. The difference represents the missing data points.
    missing_frames = list(sorted(conseq_frames - actual_frames))
    # Insert nan rows where the missing data is. Then, interpolate. 
    if len(missing_frames) > 0:
        seq = pd.DataFrame(seq)
        agent_id = seq.loc[0, 1]
        agent_type = seq.loc[0, 9]
        for missing_frame in missing_frames:
            df1 = seq[:missing_frame]
            df2 = seq[missing_frame:]
            df1.loc[missing_frame] = [
                missing_frame, agent_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
                agent_type, imputed_flag, np.nan, np.nan]
            # df1.loc[missing_frame] = [
            #     missing_frame, agent_id, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 
            #     agent_type, np.nan, np.nan]
            seq = pd.concat([df1, df2]).astype(float)
        seq = seq.interpolate(method='linear').to_numpy()[:seq_len]
    return seq

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
    max_num_agents = max([x.shape[0] for x in tensor_list]) if max_pad is None else max_pad

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
    max_num_agents = max([x.shape[0] for x in tensor_list]) if max_pad is None else max_pad

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
    dist =  np.linalg.norm(points[:, 0:2] - query_point, axis = 1) 
    topk_idxs = dist.argsort()[:k_closest]
    return points[topk_idxs]

def compute_adjacency(context, adj_type='fully_connected', dist_threshold = 0.050):
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
        x, y = rel_sequences[n, current_time_step,  0], rel_sequences[n, current_time_step, 1]
        local_context = extract_closest_points_from_query(
            points=tf_global_context, query_point=[x,y], k_closest=num_local_points)
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