from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from typing import Optional
from util.inpaint.get_mask import MaskCreator

import pathlib
import cv2
import torch
import numpy as np


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, args):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create  a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        self.args = args
        self.mask_creator = MaskCreator()

    def __call__(self, gt_im):
        mask1 = self.mask_creator.stroke_mask(self.args.image_size, self.args.image_size, max_length=self.args.image_size//2)
        mask2 = self.mask_creator.rectangle_mask(self.args.image_size, self.args.image_size, self.args.image_size//4, self.args.image_size//2)

        mask = mask1+mask2
        mask = mask > 0
        # mask = mask.astype(np.float)
        mask = mask.astype(float)
        mask = torch.from_numpy(1 - mask).unsqueeze(0)

        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        gt = (gt_im - mean[:, None, None]) / std[:, None, None]

        return gt.float()


class FFHQDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, args, big_test=False):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args
        self.big_test = big_test

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        transform = transforms.Compose([transforms.ToTensor(), DataTransform(self.args)])
        # full_data = datasets.ImageFolder(self.args.data_path, transform=transform)
        # test_data = torch.utils.data.Subset(full_data, range(50000, 70000))

        test_data = datasets.ImageFolder(self.args.data_path_test, transform=transform)

        # self.full_data, self.test_data = full_data, test_data
        self.test_data = test_data

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    # def train_dataloader(self):
    #     return DataLoader(
    #         dataset=self.full_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=4,
    #         drop_last=True,
    #         pin_memory=False
    #     )

    # def val_dataloader(self):
    #     return DataLoader(
    #         dataset=self.full_data,
    #         batch_size=self.args.batch_size,
    #         num_workers=4,
    #         drop_last=True,
    #         pin_memory=False
    #     )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.args.batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )



class FFHQ_Result_DataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, args, big_test=False):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args
        self.big_test = big_test

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        transform = transforms.Compose([transforms.ToTensor(), DataTransform(self.args)])
        full_output_data = datasets.ImageFolder(self.args.algo_out_path, transform=transform)
        # test_data = torch.utils.data.Subset(full_data, range(50000, 70000))

        test_data = datasets.ImageFolder(self.args.data_path_test, transform=transform)

        self.full_output_data, self.test_data = full_output_data, test_data

    # define your dataloaders

    def output_dataloader(self):
        return DataLoader(
            dataset=self.full_output_data,
            batch_size=1,
            num_workers=4,
            drop_last=False,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=1,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )
