from copy import deepcopy
from easydict import EasyDict
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Optional

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

    def prepare_data(self):
        # NOTE: loads a list to write into another list???
        def get_airport_files_from_list(filepath: str):
            with open(filepath, 'r') as file:
                print(f"Loading test set from {filepath}")
                airport_files = [line.strip() for line in file.readlines()]
            return airport_files

        test_list = get_airport_files_from_list(self.eparams.filepath)
        
        with open(f"{self.data_prep.in_data_dir}/test_from_list.txt", 'w') as fp:
            fp.write('\n'.join(test_list))

    def setup(self, stage: Optional[str] = None):
        """
        Processes the input data within the dataset object and randomly splits it. 

        NOTE: This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so
        be careful not to execute things like random split twice!
        """
        if not self.data_test:
            print(f"Processing testing set")
            self.data_test = deepcopy(self.dataset)
            self.data_test.set_split('test_from_list')
            self.data_test.process_data()
            print(f"...done!")

            print(f"Dataset setup complete!")

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.eparams.batch_size,
            num_workers=self.eparams.num_workers,
            pin_memory=self.eparams.pin_memory,
            shuffle=False,
            collate_fn=self.dataset.collate_batch,
        )

if __name__ == "__main__":
    _ = DataModule()