import json
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch

import amelia.scenes.utils.common as C
import src.utils.global_masks as G

from easydict import EasyDict
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple, List


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

        self.in_data_dir = config.in_data_dir
        self.context_dir = config.context_dir
        self.assets_dir = config.assets_dir

        # Trajectory configuration
        self.hist_len = config.hist_len
        self.pred_lens = config.pred_lens
        self.pred_len = max(self.pred_lens)
        self.curr_timestep = self.hist_len - 1

        self.min_agents = config.min_agents
        self.max_agents = config.max_agents

        self.add_context = config.add_context
        self.num_polylines = config.num_polylines
        self.debug = config.debug

        self.sampling_strategy = config.sampling_strategy
        self.supported_strategies = config.supported_sampling_strategies
        assert self.sampling_strategy in self.supported_strategies, \
            f"Strategy {self.sampling_strategy} not in supported list {self.supported_strategies}"
        self.k_agents = config.k_agents

        self.seed = config.seed
        self.encode_agent_type = config.encode_agent_type
        self.encode_interp_flag = config.encode_interp_flag
        self.num_agent_types = len(G.AGENT_TYPES.keys())

    def set_split_list(self, split_list: str) -> None:
        self.split_list = split_list

    # def set_blacklist(self, blacklist: dict) -> None:
    #     self.blacklist = blacklist

    # def get_blacklist(self) -> dict:
    #     return self.blacklist

    def prepare_data(self) -> None:
        raise NotImplementedError

    def collate_batch(self, batch_data):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
