#!/usr/bin/env python
# encoding: utf-8

# Data Augmentation class which is used with DataLoader
# Assume numpy array face images with B x C x H x W  [-1~1]

import scipy as sp
import numpy as np
from skimage import transform
from torchvision import transforms
from torch.utils.data import Dataset
import pdb

class FaceIdPoseDataset(Dataset):

    #  assume images  as B x C x H x W  numpy array
    def __init__(self, images, IDs, poses, transform=None):

        self.images = images
        self.IDs = IDs
        self.poses = poses
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx]
        ID = self.IDs[idx]
        pose = self.poses[idx]
        if self.transform:
            image = self.transform(image)

        return [image, ID, pose]


class Resize(object):

    #  assume image  as C x H x W  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image):
        new_h, new_w = self.output_size
        pad_width = int((new_h - image.shape[1]) / 2)
        resized_image = np.lib.pad(image, ((0,0), (pad_width,pad_width),(pad_width,pad_width)), 'edge')

        return resized_image


class RandomCrop(object):

    #  assume image  as C x H x W  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_image = image[:, top:top+new_h, left:left+new_w]

        return cropped_image
