#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
from scipy import misc
import pdb
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable


def Generate_Image(images, pose_code, Nz, G_model, args):
    """
    Generate_Image with learned Generator

    ### input
    images      : source images
    pose_code   : vector which specify pose to generate image from source image
    Nz          : size of noise vecotr
    G_model     : learned Generator
    args        : options

    ### output
    features    : extracted disentangled features of each image

    """
    if args.cuda:
        D_model.cuda()
        G_model.cuda()

    G_model.eval()

    image_size = images.shape[0]
    epoch_time = np.ceil(image_size / args.batch_size).astype(int)
    features = []
    image_number = 1

    if not(args.multi_DRGAN):

        for i in range(epoch_time):
            start = i*args.batch_size
            end = start + args.batch_size
            batch_image = torch.FloatTensor(images[start:end])
            batch_pose_code = torch.FloatTensor(pose_code[start:end])  # Condition 付に使用
            minibatch_size = len(batch_image)

            fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, Nz)))

            if args.cuda:
                batch_image, fixed_noise, batch_pose_code = \
                    batch_image.cuda(), fixed_noise.cuda(), batch_pose_cod.cuda()

            batch_image, fixed_noise, batch_pose_code = \
                Variable(batch_image), Variable(fixed_noise), Variable(batch_pose_code)

            # Generatorでイメージ生成
            generated = G_model(batch_image, batch_pose_code, fixed_noise)
            features.append(G_model.features)

            # バッチ毎に生成したイメージを
            for j in range(minibatch_size):
                save_generated_image = generated[j].cpu().data.numpy().transpose(1, 2, 0)
                save_generated_image = np.squeeze(save_generated_image)
                save_generated_image = (save_generated_image+1)/2.0 * 255.
                save_generated_image = save_generated_image[:, :, [2, 1, 0]]  # convert from BGR to RGB
                save_dir = '{}_generated'.format(args.snapshot)
                filename = os.path.join(save_dir, '{}.jpg'.format(str(image_number)))
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                print('saving {}'.format(filename))
                misc.imsave(filename, save_generated_image.astype(np.uint8))

                image_number += 1

        features = torch.cat(features)

    else:

        for i in range(epoch_time):
            start = i*args.batch_size
            end = start + args.batch_size
            batch_image = torch.FloatTensor(images[start:end])
            batch_pose_code = torch.FloatTensor(pose_code[start:end])  # Condition 付に使用
            batch_pose_code_unique = torch.FloatTensor(batch_pose_code[::args.images_perID])
            minibatch_size_unique = len(batch_image) // args.images_perID

            fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size_unique, Nz)))

            if args.cuda:
                batch_image, fixed_noise, batch_pose_code_unique = \
                    batch_image.cuda(), fixed_noise.cuda(), batch_pose_code_unique.cuda()

            batch_image, fixed_noise, batch_pose_code_unique = \
                Variable(batch_image), Variable(fixed_noise), Variable(batch_pose_code_unique)

            # Generatorでイメージ生成
            generated = G_model(batch_image, batch_pose_code_unique, fixed_noise)
            features.append(G_model.features)

            # バッチ毎に生成したイメージを
            for j in range(minibatch_size_unique):
                save_generated_image = generated[j].cpu().data.numpy().transpose(1, 2, 0)
                save_generated_image = np.squeeze(save_generated_image)
                save_generated_image = (save_generated_image+1)/2.0 * 255.
                save_generated_image = save_generated_image[:, :, [2, 1, 0]]  # convert from BGR to RGB
                save_dir = '{}_generated'.format(args.snapshot)
                filename = os.path.join(save_dir, '{}.jpg'.format(str(image_number)))
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                print('saving {}'.format(filename))
                misc.imsave(filename, save_generated_image.astype(np.uint8))

                image_number += 1

        features = torch.cat(features)
