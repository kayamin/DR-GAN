#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb

def create_multiDR_GAN_traindata(images, id_labels, pose_labels, args):
    """
    Create data set for multi-image DR-GAN

    1. remove IDs which has images less than args.images_perID
    2. sample args.images_perID images for each ID randomely
    3. order each ID's data set block randomely and return them as a training data
    """

    count = plt.hist(id_labels,np.max(id_labels))

    # 画像がn枚以上ある個人にのみ学習対象を絞る
    n = args.images_perID
    id_target = np.where(count[0]>n-1)[0]
    image_No_target = [i for i, id in enumerate(id_labels) if id in id_target]

    images_target = images[image_No_target]
    id_labels_target = id_labels[image_No_target]
    pose_labels_target = pose_labels[image_No_target]

    images_train = np.zeros((n*len(id_target), images.shape[1], images.shape[2], images.shape[3]))
    id_labels_train = np.zeros(n*len(id_target))
    pose_labels_train = np.zeros(n*len(id_target))

    # ランダムな順番で各個人からn枚ずつ画像をサンプリング
    k = 0
    for i in random.sample(list(id_target), len(id_target)):
        images_ind = images_target[id_labels_target==i]
        id_labels_ind = id_labels_target[id_labels_target==i]
        pose_labels_ind = pose_labels_target[id_labels_target==i]

        # 同一人物写真から n 枚サンプル
        sample = random.sample(range(images_ind.shape[0]), n)
        images_sample = images_ind[sample]
        id_labels_sample = id_labels_ind[sample]
        pose_labels_sample = pose_labels_ind[sample]

        images_train[k:k+n] = images_sample
        id_labels_train[k:k+n] = id_labels_sample
        pose_labels_train[k:k+n] = pose_labels_sample

        k = k+n

    return [images_train, id_labels_train.astype('int'), pose_labels_train.astype('int')]
