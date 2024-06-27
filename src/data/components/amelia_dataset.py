import numpy as np
import pickle
import random
import torch

import src.utils.data_utils as D
import src.utils.transform_utils as T
import src.utils.global_masks as G

from easydict import EasyDict
from math import radians, sin, cos
from typing import Dict

from src.data.components.base_dataset import BaseDataset

class AmeliaDataset(BaseDataset):
    """ Dataset class for post-processing the SWIM data (i.e., methods required by __getitem__()). """
    def __init__(self, config: EasyDict) -> None:
        """ Inherits base methods from BaseDataset. 

        Inputs
        ------
            config[EasyDict]: dictionary containing configuration parameters needed to process the 
            airport trajectory data. 
        """
        super(AmeliaDataset, self).__init__(config=config)
        

    def collate_batch(self, batch_data: Dict) -> Dict:
        """ Collate function prepares tensor data and adds padding where necessary. 

        Input
        -----
            batch_data[Dict]: tuple containing the scenes to prepare.
        
        Output
        ------
            batch_data[Dict]: dictionary containing the prepared batch. 
        """
        batch_size = len(batch_data)
        key_to_list = {}
        for key in batch_data[0].keys():
            key_to_list[key] = [batch_data[idx][key] for idx in range(batch_size)]

        input_dict = {}
        for key, val_list in key_to_list.items():
            if key in ['scenario_id', 'airport_id', 'ego_agent_id', 'num_agents']:
                input_dict[key] = np.asarray(val_list)
            elif key in ['sequences', 'rel_sequences']:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = D.merge_seq3d_by_padding(val_list, max_pad=self.k_agents)
            elif key in ['agent_masks']:
                val_list = [torch.from_numpy(x) for x in val_list]
                input_dict[key] = D.merge_seq2d_by_padding(val_list, max_pad=self.k_agents)
            elif key in ['context', 'adjacency']:
                input_dict[key] = None
                if not val_list[0] is None:
                    val_list = [torch.from_numpy(x).type(torch.FloatTensor) for x in val_list]
                    input_dict[key] = D.merge_seq3d_by_padding(val_list, max_pad=self.k_agents)
            else:
                val_list = [torch.from_numpy(np.asarray(x)) for x in val_list]
                input_dict[key] = D.merge_seq1d_by_padding(val_list)
            
        return {
            'batch_size': batch_size, 'scene_dict': input_dict, 'strategy': self.sampling_strategy
        }

    def transform_sequences(self, sequences: np.array, ego_agent_id: int = 0) -> np.array:
        """ Transforms the scene w.r.t. the ego_agent's reference frame. 

        Inputs
        ------
            sequences[np.array]: numpy array containing scene information in absolute coordinates. 
            ego_agent_id[int]: index of the agent chosen to be the ego-agent (NOTE: currently, this
            is done randomly during sharding.)
        
        Output
        ------
            rel_seq[np.array]: numpy array containing the trajectory sequences in the ego-agent's 
            reference frame.
        """
        num_agents, timesteps, _ = sequences.shape
        rel_sequence = np.zeros(shape=(num_agents, timesteps, 4)) # [x, y, z, heading]
    
        # the ego-agent's heading at the 'current time step'
        ego_heading = radians(sequences[ego_agent_id, self.curr_timestep, G.SEQ_IDX.Heading])

        R = np.array(
            [[cos(ego_heading), -sin(ego_heading), 0.0],
             [sin(ego_heading),  cos(ego_heading), 0.0],
             [             0.0,               0.0, 1.0]])
        R = np.repeat(R.reshape(1, 3, 3), num_agents, axis=0)

        rel_xyz = sequences[:, :, G.XYZ] - sequences[ego_agent_id, self.curr_timestep, G.XYZ]
        rel_sequence[:, :, :3] = np.matmul(rel_xyz, R)

        headings = sequences[:, :, G.SEQ_IDX.Heading]
        ego_heading = sequences[ego_agent_id, self.curr_timestep, G.SEQ_IDX.Heading]

        # wrap the angle
        rel_sequence[:, :, -1] = T.wrap_angle(headings - ego_heading)
        return rel_sequence

    def transform_context(
        self, semantic_map: np.array, sequences: np.array, rel_sequences: np.array, ego_agent: int, 
        limits: list
    ) -> np.array:
        """ Generates the map context for a given sequence and ego agent ID. Using the polyline 
        representaton of the map. 
            
        Inputs
        ------
            semantic_map[np.array]: numpy array containing the vectorized representation of the map.
            sequences[np.array]: numpy array with the scene information in global frame.
            rel_sequences[np.array]: numpy array with the scene information in the ego-agent's frame.
            ego_agent[float]: index of the ego agent. 
            limits[list]: list containing the global limits in latitude and longitude of the map. 

        Outputs
        -------
            semantic_map[np.array(N, #Polylines, D = 11)]: the corresponding scene's map information
            in relative frame.
        """
        ego_position = sequences[ego_agent, self.curr_timestep, G.XY]
        ego_heading = radians(sequences[ego_agent, self.curr_timestep, G.SEQ_IDX.Heading])
        semantic_map, adjacency = D.compute_local_context_from_ego_agent(
            semantic_map, ego_position, ego_heading, rel_sequences, self.curr_timestep, 
            self.num_polylines, self.debug, ego_id=ego_agent, limits=limits
        )    
        return semantic_map, adjacency

    def transform_scene_data(self, scene_data: Dict, seed: int = 42, random_ego:bool = True) -> Dict:
        """ Transforms scene's global data to the ego-agent's reference frame. 
        
        Input
        -----
            scene_data[Dict]: a dictionary containing the pre-processed scene information.
        
        Output
        ------
            scene_dict[Dict]: the transformed scene data. 
        """
        sequences = scene_data['sequences']
        agent_masks = scene_data['agent_masks']
        airport_id = scene_data['airport_id']

        # TODO: fix this k-agent selection. This is assuming that the k selected agents are related
        #       or relevant to each other, but this is not a guarantee. They could be complete 
        #       unrelated, and thus the scene representation may not be valid. 
        # Define ego agent and agents in scene based on the sampling scheme
        # if self.sampling_strategy == 'random':
        #     # For k_random, slice k agents out of previously shuffled agent array
        #     agents_in_scene = scene_data['random_order'][:self.k_agents]
        # elif self.sampling_strategy == 'safety':
        #     # For safety oriented, splice k agents out of ordered agent array 
        #     agents_in_scene = scene_data['critical_order'][:self.k_agents]
        # else:
        #     raise ValueError(f"Sampling strategy: {self.sampling_strategy} not supported!")
        agents_in_scene = scene_data['meta']['agent_order'][self.sampling_strategy][:self.k_agents]
        
        # Choose an ego-agent from the valid ones. NOTE: Valid ones are should appear first. 
        # num_agents = min(self.k_agents, 2)#scene_data['random_valid'])
        num_agents = len(agents_in_scene)
        # TODO: get GLOBAL seed from config files
        # random.seed(seed)
        if random_ego:
            ego_agent = random.randint(a=0, b=num_agents-1)
        else:
            ego_agent = 0 

        # Slice the number of agents from the sequence and define random ego agent
        sequences = sequences[agents_in_scene]
        agent_masks = agent_masks[agents_in_scene]

        rel_sequences = self.transform_sequences(sequences, ego_agent)

        if self.encode_interp_flag:
            rel_sequences = np.concatenate((rel_sequences, agent_masks[..., None]), axis=-1)
        
        if self.encode_agent_type:
            agent_types = np.asarray(scene_data['agent_types'])
            agent_types = agent_types[agents_in_scene, : ,:]
            index = np.arange(agent_types.size)
            agent_types_onehot = np.zeros(shape=(agent_types.shape[0], 1, self.num_agent_types))
            agent_types_onehot[index, 0, agent_types] = 1
            agent_types_onehot = np.tile(agent_types_onehot, (1, self.seq_len, 1))
            rel_sequences = np.concatenate((rel_sequences, agent_types_onehot), axis=-1)

        # TODO: debug
        context_map, adjacency = None, None
        if self.add_context:
            context_map, adjacency = self.transform_context(
                self.semantic_maps[airport_id], sequences, rel_sequences, ego_agent, 
                self.limits[airport_id]
            )
        agent_types = np.asarray(scene_data['agent_types'])
        agent_types = agent_types[agents_in_scene]
        
        return {
            'scenario_id': scene_data['scenario_id'],
            'airport_id': airport_id,
            'agent_ids': scene_data['agent_ids'], 
            'agent_types': agent_types,
            'agent_masks': agent_masks,
            'ego_agent_id': ego_agent,
            'num_agents': sequences.shape[0],
            'sequences': sequences,
            'rel_sequences': rel_sequences,
            'agents_in_scene': agents_in_scene,
            'context': context_map,
            'adjacency': adjacency,
        }
    
    def __len__(self):
        return len(self.scenario_list)

    def __getitem__(self, index):
        """ Loads scene from given index and transforms it w.r.t. to its corresponding ego-agent."""
        with open(self.scenario_list[index],'rb') as f:
            data = pickle.load(f)
        
        data = self.transform_scene_data(data)
        
        return data