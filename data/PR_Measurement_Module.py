import torch
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl


class PRMeasurementDataset(Dataset):
    def __init__(self, measurement_data_file, sig_y_file):
        self.y_data = torch.load(measurement_data_file)
        self.sig_y = torch.load(sig_y_file)

    def __len__(self):
        # Return the total number of samples
        return len(self.y_data)

    def __getitem__(self, idx):
        # Retrieve a sample and its corresponding label
        sample = self.y_data[idx]
        sig_y_sample = self.sig_y[idx]
        return sample, sig_y_sample


class PRDataModule(pl.LightningDataModule):
    """
    DataModule used for PR measurement
    """

    def __init__(self, args, big_test=False):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args

    def prepare_data(self):
        pass

    def setup(self):

        self.test_data = PRMeasurementDataset(self.args.measurement_data_file, self.args.sig_y_file)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )