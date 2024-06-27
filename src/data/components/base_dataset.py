import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch

import amelia_scenes.utils.common as C
import src.utils.global_masks as G

from easydict import EasyDict
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple, List

from amelia_scenes.scene_scoring.src.crowdedness import compute_simple_scene_crowdedness
from amelia_scenes.scene_scoring.src.kinematic import compute_kinematic_scores
from amelia_scenes.scene_scoring.src.interactive import compute_interactive_scores
from amelia_scenes.scene_scoring.src.critical import compute_simple_scene_critical

from src.utils.data_utils import impute
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)

os.environ["MKL_NUM_THREADS"] = "1"
os.system('taskset -p --cpu-list 0-128 %d' % os.getpid())
torch.multiprocessing.set_sharing_strategy('file_system')

class BaseDataset(Dataset):
    """ Dataset class for pre-processing airport surface movement data (i.e., methods required for 
    sharding the data). """
    def __init__(self, config: EasyDict) -> None:
        """
        Inputs
        ------
            config[EasyDict]: dictionary containing configuration parameters needed to process the 
            airport trajectory data. 
        """
        super(BaseDataset, self).__init__()

        # Trajectory configuration
        self.hist_len = config.hist_len
        self.pred_lens = config.pred_lens
        self.pred_len = max(self.pred_lens)
        self.curr_timestep = self.hist_len-1
        self.seq_len = self.hist_len + self.pred_len
        self.skip = config.skip
        self.min_agents = config.min_agents
        self.max_agents = config.max_agents
        self.parallel = config.parallel
        self.debug = config.debug
        self.num_polylines = config.num_polylines
        self.add_context = config.add_context
        self.in_data_dir = config.in_data_dir
        self.out_data_dir = config.out_data_dir
        self.context_dir = config.context_dir
        self.assets_dir = config.assets_dir
        self.do_sharding = config.do_sharding
        self.sampling_strategy = config.sampling_strategy
        self.k_agents = config.k_agents 
        self.split = config.split
        self.min_valid_points = 2
        self.seed = 42
        self.blacklist = {}
        os.makedirs(self.out_data_dir, exist_ok=True)        

        self.encode_agent_type = config.encode_agent_type
        self.encode_interp_flag = config.encode_interp_flag
        self.num_agent_types = len(G.AGENT_TYPES.keys())
        
    def set_split(self, split: str) -> None:
        self.split = split
    
    def set_blacklist(self, blacklist: dict) -> None:
        self.blacklist = blacklist

    def get_blacklist(self) -> dict:
        return self.blacklist

    def prepare_data(self) -> None:
        """ Prepares data for sharding: loads the graphs and limit files, and prepares output 
        directories and input files. 
        """
        log.info("Preparing data for processing.")
        # NOTE: previously removed blacklisted elements here, but now being done in DataModule.  
        # NOTE: also, overly complicated data prep process. Really need to do it in another repo 
        #       and keep this one simple. 
        with open(os.path.join(self.in_data_dir, f"{self.split}.txt")) as f:
            file_list = f.read().splitlines()

        self.out_dirs = {}
        self.semantic_maps = {}
        self.semantic_pkl = {}
        self.limits = {}
        self.ref_data = {}
        self.scenario_list = {}
        self.hold_lines = {}
        self.data_files = []

        airports = list(set([f.split('/')[0] for f in file_list]))        
        for airport in airports:
            graph_file = os.path.join(self.context_dir, airport, 'semantic_graph.pkl')
            with open(graph_file, 'rb') as f:
                temp_dict = pickle.load(f)
                self.semantic_pkl[airport] = temp_dict
                self.semantic_maps[airport] = temp_dict['map_infos']['all_polylines'][:, G.MAP_IDX]
                self.hold_lines[airport] = temp_dict['hold_lines']
        
            limits_file = os.path.join(self.assets_dir, airport, 'limits.json')
            with open(limits_file, 'r') as fp:
                self.ref_data[airport] = EasyDict(json.load(fp))

            self.limits[airport] = (
                self.ref_data[airport].espg_4326.north, 
                self.ref_data[airport].espg_4326.east, 
                self.ref_data[airport].espg_4326.south, 
                self.ref_data[airport].espg_4326.west
            )

            self.out_dirs[airport] = os.path.join(self.out_data_dir, airport)
            os.makedirs(self.out_dirs[airport], exist_ok=True)

            self.scenario_list[airport] = []
        
        self.data_files = [os.path.join(self.in_data_dir, f) for f in file_list]

    def process_data(self) -> None:
        """ If do_sharding is True, it will process the CSV data containing airport trajectory 
        information and create shards containing scenario-level pickle data. If parallel is True, it 
        will process the data in parallel, otherwise it will do it sequentially. Once the sharding 
        is done, it will return a list containing all generated scenarios. 
        
        If do_sharding is False, it assumes the shards have been created already and it will only 
        return the list of available scenarios. 
        """
        self.prepare_data()
        log.info(f"Processing data.")

        # Process files
        if self.parallel:  
            scenarios = Parallel(n_jobs=32)(
                delayed(self.process_file)(f) for f in tqdm(self.data_files))
            # Unpacking results
            for i in range(len(scenarios)):
                res = scenarios.pop()
                if res is not None:
                    airport = res[-1]
                    self.scenario_list[airport] += res[0]
                    if res[1] is not None: 
                        self.blacklist[airport] += res[1]
            del scenarios
        else:
            for f in tqdm(self.data_files):
                res = self.process_file(f)
                if res is not None:
                    airport = res[-1]
                    self.scenario_list[airport] += res[0]
                    if res[1] is not None: 
                        self.blacklist[airport] += res[1]
        
        # TODO: find cleaner solution to this: 
        # Balance the number of scenarios in the multi-airport setting. For some airports, scenario 
        # rejection is more marked, but for a 'fair' comparison I'm trying to keep the balance 
        # between airport. Also, if an airport gets too many rejections the number of files that get 
        # in per airport will depend on it, which is also not great. 
        if len(self.scenario_list.keys()) > 0:    
            files_per_airport = max(min(len(f) for _, f in self.scenario_list.items()), 500000)
            balanced_list = []
            for airport, files in self.scenario_list.items():
                # TODO: need seed 
                random.shuffle(files)
                balanced_list += files[:files_per_airport]
            self.scenario_list = balanced_list

        # Once all of the data has been processed and the blacklists collected, save them.
        blacklist_dir = os.path.join(self.in_data_dir, 'blacklist')
        split_type = self.split.split('_')[1]
        for airport, airport_blacklist in self.blacklist.items():
            blacklist_file = os.path.join(blacklist_dir, f'{airport}_{split_type}.txt')
            with open(blacklist_file, 'w') as fp:
                fp.write('\n'.join(airport_blacklist))

    def process_file(self, f: str) -> Tuple[List, List, List, List, List, List]:
        """ Processes a single data file. It first obtains the number of possible sequences (given 
        the parameters in the configuration file) and then generates scene-level pickle files with
        the corresponding scene's information. 

        Inputs
        ------
            f[str]: name of the file to shard. 
        """
        shard_name = f.split('/')[-1].split('.')[0]
        airport_id = f.split('/')[-1].split('_')[0].lower()
        data_dir = os.path.join(self.out_dirs[airport_id], shard_name)

        # Check if the file has been sharded already. If so, add sharded files to the scenario list.
        if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
            pkl_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
            return pkl_list, None, airport_id
        
        # Otherwise, shard the file and add it to the scenario list. 
        data = pd.read_csv(f)

        # Get the number of unique frames
        frames = data.Frame.unique().tolist()
        frame_data = []
        for frame_num in frames:
            frame = data[:][data.Frame == frame_num] 
            frame_data.append(frame)

        num_sequences = int(math.ceil((len(frames) - (self.seq_len) + 1) / self.skip))
        if num_sequences < 1:
            return

        sharded_files = []
        blacklist = []
        os.makedirs(data_dir, exist_ok=True)
        
        valid_seq = 0
        for i in range(0, num_sequences * self.skip + 1, self.skip):
            scenario_id = str(valid_seq).zfill(5)
            seq, agent_id, agent_type, agent_valid, agent_mask = self.process_seq(
                frame_data=frame_data, frames=frames, seq_idx=i, airport_id=airport_id)
            if seq is None:
                continue

            # Get agent array based on random and safety criteria
            num_agents, _, _ = seq.shape 

            # NOTE: an overkill way to create a randomized list of valid agent indeces + interpolated
            # agent indeces. This is so that the __getitem__ function can choose a random ego-agent. 
            agents_in_scene = np.asarray(list(range(num_agents)))
            random.shuffle(agents_in_scene)
            random_agents_in_scene = np.asarray(agents_in_scene)

            # TODO: needs seed 
            valid_agents = agents_in_scene[agent_valid]
            # random.shuffle(valid_agents)
            num_valid = len(valid_agents)
            # invalid_agents = agents_in_scene[~agent_valid]
            # random.shuffle(invalid_agents)
            # random_agents_in_scene = np.asarray(valid_agents.tolist() + invalid_agents.tolist())

            # Get random agents in scene using as weights safety score
            # agents_scores, scene_score = get_safety_score(
            #     sequences=seq, agent_types=agent_type, graph=self.semantic_pkl[airport_id], 
            #     airport=airport_id, ref_data=self.ref_data[airport_id])
            # sorted_indices = np.argsort(agents_scores)[::-1]
            # # TODO: how to handle valid agents and critical scoring. Need to make sure only valid 
            # #       agents are selected as most critical. 
            # #       NOTE: set intersection sorts elements so doing it manually:
            # critical_agents_in_scene = np.asarray(
            #     [i for i in sorted_indices if i in valid_agents] +
            #     [i for i in sorted_indices if i not in valid_agents])
            
            scenario = {
                'scenario_id': scenario_id, 
                'sequences': seq, 
                'num_agents': num_agents,
                'random_order': random_agents_in_scene,
                'random_valid': num_valid,
                # 'critical_order': sorted_indices,
                # 'critical_order_sorted': critical_agents_in_scene,
                # 'agents_score': agents_scores,
                # 'scene_score': scene_score,
                'agent_sequences': seq.copy(), 
                'agent_ids': agent_id, 
                'agent_types': agent_type,
                'agent_masks': agent_mask,
                'agent_valid': agent_valid,
                'airport_id': airport_id,
            }

            crowd_scene_score = compute_simple_scene_crowdedness(
                scene=EasyDict(scenario), 
                max_agents=self.max_agents
            )
            kin_agents_scores, kin_scene_score = compute_kinematic_scores(
                scene=EasyDict(scenario), 
                hold_lines=self.hold_lines[airport_id]
            )
            int_agents_scores, int_scene_score = compute_interactive_scores(
                scene=EasyDict(scenario), 
                hold_lines=self.hold_lines[airport_id]
            )
            crit_agent_scores, crit_scene_score =  compute_simple_scene_critical(
                agent_scores_list=[kin_agents_scores, int_agents_scores],
                scene_score_list=[crowd_scene_score, kin_scene_score, int_scene_score]
            )

            scenario['meta'] = {
                'agent_scores': {
                'kinematic': kin_agents_scores,
                'interactive': int_agents_scores, 
                'critical': crit_agent_scores
                },
                'agent_order': {
                    'random': C.get_random_order(num_agents, agent_valid, self.seed),
                    'kinematic': C.get_sorted_order(kin_agents_scores),
                    'interactive': C.get_sorted_order(int_agents_scores), 
                    'critical': C.get_sorted_order(crit_agent_scores)
                },
                'scene_scores': {
                    'crowdedness': crowd_scene_score,
                    'kinematic': kin_scene_score,
                    'interactive': int_scene_score, 
                    'critical': crit_scene_score
                },
            }

            scenario_filepath = os.path.join(data_dir, f"{scenario_id}.pkl")
            with open(scenario_filepath,'wb') as f:
                pickle.dump(scenario, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            valid_seq += 1
            sharded_files.append(scenario_filepath)
        
        # If directory is empty, remove it.
        if len(os.listdir(data_dir)) == 0:
            blacklist.append(f.removeprefix(self.in_data_dir))
            os.rmdir(data_dir)
        return sharded_files, blacklist, airport_id

    def process_seq(
        self, frame_data: pd.DataFrame, frames: list, seq_idx: int, airport_id: str
    ) -> np.array:
        """ Processes all valid agent sequences.

        Inputs:
        -------
            frame_data[pd.DataFrame]: dataframe containing the scene's trajectory information in the 
            following format:
                <FrameID, AgentID, Altitude, Speed, Heading, Lat, Lon, Range, Bearing, AgentType, 
                 Interp, x, y>
            frames[list]: list of frames to process. 
            seq_idx[int]: current sequence index to process.  
        
        Outputs:
        --------
            seq[np.array]: numpy array containing all processed scene's sequences
            agent_id_list[list]: list with the agent IDs that were processed.
            agent_type_list[list]: list containing the type of agent (Aircraft = 0, Vehicle = 1, 
            Unknown=2)
        """
        none_outs = (None, None, None, None, None)
        # All data for the current sequence: from the curr index i to i + sequence length
        seq_data = np.concatenate(frame_data[seq_idx:seq_idx + self.seq_len], axis=0)

        # IDs of agents in the current sequence
        unique_agents = np.unique(seq_data[:, G.RAW_IDX.ID])
        num_agents = len(unique_agents)
        if num_agents < self.min_agents or num_agents > self.max_agents:
            return none_outs

        num_agents_considered = 0
        seq = np.zeros((num_agents, self.seq_len, G.DIM))
        agent_masks = np.zeros((num_agents, self.seq_len)).astype(bool)
        agent_id_list, agent_type_list, valid_agent_list = [], [], []
    
        alt_idx = G.RAW_IDX.Altitude

        for _, agent_id in enumerate(unique_agents):
            # Current sequence of agent with agent_id
            agent_seq = seq_data[seq_data[:, 1] == agent_id]

            # Start frame for the current sequence of the current agent reported to 0
            pad_front = frames.index(agent_seq[0, 0]) - seq_idx
            
            # End frame for the current sequence of the current agent: end of current agent path in 
            # the current sequence. It can be sequence length if the aircraft appears in all frames
            # of the sequence or less if it disappears earlier.
            pad_end = frames.index(agent_seq[-1, 0]) - seq_idx + 1
            
            # Exclude trajectories less then seq_len
            if pad_end - pad_front != self.seq_len:
                continue
            
            # Scale altitude 
            mx = self.ref_data[airport_id].limits.Altitude.max
            mn = self.ref_data[airport_id].limits.Altitude.min
            agent_seq[:, alt_idx] = (agent_seq[:, alt_idx] - mn) / (mx - mn)

            agent_id_list.append(int(agent_id))
            # TODO: fix this. Agent type is not necessarily fixed for the entire trajectory.
            agent_type_list.append(int(agent_seq[0, G.RAW_IDX.Type]))

            # Interpolated mask
            mask = agent_seq[:, G.RAW_IDX.Interp] == '[ORG]'
            agent_seq[mask, G.RAW_IDX.Interp]  = 1.0 # Not interpolated -->     Valid
            agent_seq[~mask, G.RAW_IDX.Interp] = 0.0 #     Interpolated --> Not valid
            
            # Check if there's at least two valid points in the history segment, two valid points in 
            # partial segment and two valid points in the future segment
            valid = mask[:self.hist_len].sum() >= self.min_valid_points
            if valid:
                for t in self.pred_lens:
                    if mask[self.hist_len:self.hist_len+t].sum() < self.min_valid_points:
                        valid = False
                        break
            valid_agent_list.append(valid)

            # TODO: debug impute
            # Impute missing data using linear interpolation 
            agent_seq = impute(agent_seq, self.seq_len)
            valid_mask = agent_seq[:, G.RAW_IDX.Interp].astype(bool)
            agent_masks[num_agents_considered, pad_front:pad_end] = valid_mask

            agent_seq = agent_seq[:, G.RAW_SEQ_MASK]
            seq[num_agents_considered, pad_front:pad_end] = agent_seq[:, G.SEQ_ORDER]
            num_agents_considered += 1
        
        # Return Nones if there aren't any valid agents
        valid_agent_list = np.asarray(valid_agent_list)
        if valid_agent_list.sum() == 0:
            return none_outs

        # Return Nones if the number of considered agents is less than the required
        if num_agents_considered < self.min_agents:
            return none_outs
        
        return seq[:num_agents_considered], agent_id_list, agent_type_list, valid_agent_list, \
            agent_masks[:num_agents_considered]
    
    def collate_batch(self, batch_data):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError