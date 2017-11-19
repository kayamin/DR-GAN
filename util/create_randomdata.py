#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
from torch.autograd import Variable

def create_randomdata(data_size=300, channel_num=3, Nd=5, Np=9):
    """
    Create random data

    ### ouput
    images : 4 dimension tensor (the number of image x channel x image_height x image_width)
    id_labels : one-hot vector with Nd dimension
    pose_labels : one-hot vetor with Np dimension
    Nd : the nuber of ID in the data
    Np : the number of discrete pose in the data
    Nz : size of noise vector
    """
    Nz = 50
    images = np.random.randn(data_size, channel_num, 110,110)
    id_labels = np.random.randint(Nd, size=data_size)
    pose_labels = np.random.randint(Np, size=data_size)

    return [images, id_labels, pose_labels, Nd, Np, Nz, channel_num]
