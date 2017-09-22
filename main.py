#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import datetime
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from model import single_DR_GAN_model as single_model
from model import multiple_DR_GAN_model as multi_model
from util.create_randomdata import create_randomdata
from train import train
from Generate_Image import Generate_Image
import pdb


def DataLoader():
    """
    Define dataloder which is applicable to your data

    ### ouput
    images : 4 dimension tensor (the number of image x channel x image_height x image_width)
    id_labels : one-hot vector with Nd dimension
    pose_labels : one-hot vetor with Np dimension
    Nd : the nuber of ID in the data
    Np : the number of discrete pose in the data
    Nz : size of noise vector
    """
    Nd = []
    Np = []
    Nz = []
    channel_num = []
    images = []
    id_labels = []
    pose_labels = []

    return [images, id_labels, pose_labels, Nd, Np, Nz, channel_num]


if __name__=="__main__":

    # argparse を用いてコマンドライン引数を解析
    parser = argparse.ArgumentParser(description='DR_GAN')
    # learning
    parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
    parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch_size', type=int, default=3, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    # data
    parser.add_argument('-random', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-multi-DRGAN', action='store_true', default=False, help='use multi image DR_GAN model')
    parser.add_argument('-images-perID', type=int, default=0, help='number of images per person to input to multi image DR_GAN')
    parser.add_argument('-cuda', action='store_true', default=False, help='enable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot(snaphsot/{date}/{epoch}) [default: None]')
    parser.add_argument('-generate', action='store_true', default=None, help='Generate pose modified image from given image')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    args = parser.parse_args()

    # update args and print
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))

    # input data
    if args.random:
        images, id_labels, pose_labels, Nd, Np, Nz, channel_num = create_randomdata()
    else:
        images, id_labels, pose_labels, Nd, Np, Nz, channel_num = DataLoader()

    # model
    if args.snapshot is None:
        if not(args.multi_DRGAN):
            D = single_model.Discriminator(Nd, Np, channel_num)
            G = single_model.Generator(Np, Nz, channel_num)
        else:
            if args.images_perID==0:
                print("Please specify -images-perID of your data to input to multi_DRGAN")
                exit()
            else:
                D = multi_model.Discriminator(Nd, Np, channel_num)
                G = multi_model.Generator(Np, Nz, channle_num, args.images_perID)
    else:
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            D = torch.load('{}_D.pt'.format(args.snapshot))
            G = torch.load('{}_G.pt'.format(args.snapshot))
        except:
            print("Sorry, This snapshot doesn't exist.")
            exit()

    if not(args.generate):
        train(images, id_labels, pose_labels, Nd, Np, Nz, D, G, args)
    else:
        pose_code = [] # specify arbitrary pose code for every image
        # pose_code = np.random.uniform(-1,1, (images.shape[0], Np))

        features = Generate_Image(images, pose_code, Nz, G, args)
