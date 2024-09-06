from copy import deepcopy
from re import split
from easydict import EasyDict
from lightning import LightningDataModule
from math import floor
from torch.utils.data import DataLoader, Dataset
from typing import Optional

from amelia_tf.utils import pylogger
from amelia_tf.utils import data_utils as D

log = pylogger.get_pylogger(__name__)


class DataModule(LightningDataModule):
    """ DataModule wrapper based on: lightning.ai/docs/pytorch/stable/data/datamodule.html """

    def __init__(self, dataset: Dataset, extra_params: EasyDict):
        """
        Inputs
        ------
            dataset[Dataset]: pytorch dataset object.
            extra_params[EasyDict]: dictionary containing any additional parameters needed by the
                class. NOTE: This is used in order to avoid adding more init parameters.
        """
        super().__init__()

        # This line allows to access init parameters with 'self.hparams' attribute also ensures init
        # parameterss will be stored in ckpt.
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.dataset = dataset

        self.eparams = extra_params
        self.data_prep = self.eparams.data_prep
        self.supported_airports = self.eparams.supported_airports

        self.task_name = self.eparams.task_name

        self.splits_suffix = f"{self.data_prep.split_type}_{self.data_prep.exp_suffix}"

        self.split_path = {
            "train": f"{self.data_prep.traj_data_dir}/splits/train_{self.splits_suffix}.txt",
            "val": f"{self.data_prep.traj_data_dir}/splits/val_{self.splits_suffix}.txt",
            "test": f"{self.data_prep.traj_data_dir}/splits/test_{self.splits_suffix}.txt"
        }

        assert self.task_name in self.eparams.task_names

    def prepare_data(self):
        """ Creates the data splits for training, validation and testing. """

        # NOTE: All inside this function SHOULD be done outside this repository.
        # TODO: Need to create 'amelia_dataset' repo to do data-preparation stuff. The data module
        # should only load the splits without having to deal with any of this.

        # ------------------------------------------------------------------------------------------
        # log.info("Creating dataset splits.")
        assert self.data_prep.split_type in self.eparams.supported_splits, \
            f"Data split type {self.data_prep.split_type} not in supported splits: {self.eparams.supported_splits}."

        # ------------------------------------------------------------------------------------------
        # Process 'seen' airports into train/val/test splits.
        log.info("Preparing dataset splits.")
        seen_airports = self.data_prep.seen_airports
        assert len(seen_airports) > 0, f"Train airport list is empty: {seen_airports}"
        assert len(seen_airports) == len(set(seen_airports)), f"Duplicate airports {seen_airports}"
        assert all(airport in self.supported_airports for airport in seen_airports), \
            f"Unsupported airport. Supported ones are {self.supported_airports}"

        train_list, val_list, test_list = [], [], []
        for airport in seen_airports:
            filename = f"{airport}_{self.data_prep.split_type}"
            with open(f"{self.data_prep.traj_data_dir}/splits/train_splits/{filename}.txt", 'r') as fp:
                airport_list = [line.rstrip() for line in fp]
                train_list += airport_list[:int(len(airport_list) * self.data_prep.to_process)]

            with open(f"{self.data_prep.traj_data_dir}/splits/val_splits/{filename}.txt", 'r') as fp:
                airport_list = [line.rstrip() for line in fp]
                val_list += airport_list[:int(len(airport_list) * self.data_prep.to_process)]

            with open(f"{self.data_prep.traj_data_dir}/splits/test_splits/{filename}.txt", 'r') as fp:
                airport_list = [line.rstrip() for line in fp]
                test_list += airport_list[:int(len(airport_list) * self.data_prep.to_process)]

        # ------------------------------------------------------------------------------------------
        # If 'unseen' airports are specified, then it will first iterate over unseen_airports and
        # create a random test split for each and add it to the test list.
        unseen_airports = self.data_prep.unseen_airports
        if len(unseen_airports) > 0:
            assert all(not airport in seen_airports for airport in unseen_airports), \
                f"'Unseen' airport {airport} is in 'Seen' list {seen_airports}"
            assert all(airport in self.supported_airports for airport in unseen_airports), \
                f"Unsupported airport {airport}. Supported ones are {self.supported_airports}"

            for airport in unseen_airports:
                filename = f"{airport}_{self.data_prep.split_type}"
                with open(f"{self.data_prep.traj_data_dir}/splits/test_splits/{filename}.txt", 'r') as fp:
                    airport_list = [line.rstrip() for line in fp]
                    test_list += airport_list[:int(len(airport_list) * self.data_prep.to_process)]

        # ------------------------------------------------------------------------------------------
        # Load blacklist and remove files in blacklist from split files
        self.blacklist = D.load_blacklist(self.data_prep, self.supported_airports)
        flat_blacklist = D.flatten_blacklist(self.blacklist)

        # ------------------------------------------------------------------------------------------
        # Save 'temporary' train/val/test splits.
        # TODO: verify that split lists do not share information
        self.train_list = D.remove_blacklisted(flat_blacklist, train_list)
        with open(self.split_path["train"], 'w') as fp:
            fp.write('\n'.join(self.train_list))

        self.val_list = D.remove_blacklisted(flat_blacklist, val_list)
        with open(self.split_path["val"], 'w') as fp:
            fp.write('\n'.join(self.val_list))

        self.test_list = D.remove_blacklisted(flat_blacklist, test_list)
        with open(self.split_path["test"], 'w') as fp:
            fp.write('\n'.join(self.test_list))

    def setup(self, stage: Optional[str] = None):
        """
        Processes the input data within the dataset object and randomly splits it.

        NOTE: This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so
        be careful not to execute things like random split twice!
        """
        if not self.data_train and not self.data_val and not self.data_test:
            if self.task_name == "train":
                log.info(f"Processing train set")
                self.data_train = deepcopy(self.dataset)
                self.data_train.set_split_list(self.split_path["train"])
                # self.data_train.set_blacklist(self.blacklist)
                self.data_train.prepare_data()
                log.info(f"...done!")

                log.info(f"Processing validation set")
                self.data_val = deepcopy(self.dataset)
                self.data_val.set_split_list(self.split_path["val"])
                # self.data_val.set_blacklist(self.data_train.get_blacklist())
                self.data_val.prepare_data()
                log.info(f"...done!")

            log.info(f"Processing test set")
            self.data_test = deepcopy(self.dataset)
            self.data_test.set_split_list(self.split_path["test"])
            # self.data_test.set_blacklist(self.data_val.get_blacklist())
            self.data_test.prepare_data()
            log.info(f"...done!")
            log.info(f"Dataset setup complete!")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.eparams.batch_size,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            shuffle=True,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers,
            prefetch_factor=4
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.eparams.batch_size,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            shuffle=False,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.eparams.batch_size,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            shuffle=False,
            collate_fn=self.dataset.collate_batch,
            persistent_workers=self.eparams.persistent_workers
        )


if __name__ == "__main__":
    _ = DataModule()
