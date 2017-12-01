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
from torch.utils.data import DataLoader
from torchvision import transforms
from util.one_hot import one_hot
from util.Is_D_strong import Is_D_strong
from util.log_learning import log_learning
from util.convert_image import convert_image
from util.DataAugmentation import FaceIdPoseDataset, Resize, RandomCrop


def train_single_DRGAN(images, id_labels, pose_labels, Nd, Np, Nz, D_model, G_model, args):
    if args.cuda:
        D_model.cuda()
        G_model.cuda()

    D_model.train()
    G_model.train()

    lr_Adam = args.lr
    beta1_Adam = args.beta1
    beta2_Adam = args.beta2

    image_size = images.shape[0]
    epoch_time = np.ceil(image_size / args.batch_size).astype(int)

    optimizer_D = optim.Adam(D_model.parameters(), lr=lr_Adam, betas=(beta1_Adam, beta2_Adam))
    optimizer_G = optim.Adam(G_model.parameters(), lr=lr_Adam, betas=(beta1_Adam, beta2_Adam))
    loss_criterion = nn.CrossEntropyLoss()
    loss_criterion_gan = nn.BCEWithLogitsLoss()

    loss_log = []
    steps = 0

    flag_D_strong = False
    for epoch in range(1, args.epochs+1):

        # Load augmented data
        transformed_dataset = FaceIdPoseDataset(images, id_labels, pose_labels,
                                                transform=transforms.Compose([Resize((110, 110)), RandomCrop((96, 96))]))
        dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True)

        for i, batch_data in enumerate(dataloader):
            D_model.zero_grad()
            G_model.zero_grad()

            batch_image = torch.FloatTensor(batch_data[0].float())
            batch_id_label = batch_data[1]
            batch_pose_label = batch_data[2]
            minibatch_size = len(batch_image)

            batch_ones_label = torch.ones(minibatch_size)   # 真偽判別用のラベル
            batch_zeros_label = torch.zeros(minibatch_size)

            # ノイズと姿勢コードを生成
            fixed_noise = torch.FloatTensor(np.random.uniform(-1, 1, (minibatch_size, Nz)))
            tmp = torch.LongTensor(np.random.randint(Np, size=minibatch_size))
            pose_code = one_hot(tmp, Np)  # Condition 付に使用
            pose_code_label = torch.LongTensor(tmp)  # CrossEntropy 誤差に使用

            if args.cuda:
                batch_image, batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_pose_label.cuda(), batch_ones_label.cuda(), batch_zeros_label.cuda()

                fixed_noise, pose_code, pose_code_label = \
                    fixed_noise.cuda(), pose_code.cuda(), pose_code_label.cuda()

            batch_image, batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label = \
                Variable(batch_image), Variable(batch_id_label), Variable(batch_pose_label), Variable(batch_ones_label), Variable(batch_zeros_label)

            fixed_noise, pose_code, pose_code_label = \
                Variable(fixed_noise), Variable(pose_code), Variable(pose_code_label)

            # Generatorでイメージ生成
            generated = G_model(batch_image, pose_code, fixed_noise)

            steps += 1

            # バッチ毎に交互に D と G の学習，　Dが90%以上の精度の場合は 1:4の比率で学習
            if flag_D_strong:

                if i % 5 == 0:
                    # Discriminator の学習
                    flag_D_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
                                            batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)

                else:
                    # Generatorの学習
                    Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G, generated,\
                            batch_id_label, batch_ones_label, pose_code_label, epoch, steps, Nd, args)
            else:

                if i % 2 == 0:
                    # Discriminator の学習
                    flag_D_strong = Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
                                            batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args)

                else:
                    # Generatorの学習
                    Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G, generated, \
                            batch_id_label, batch_ones_label, pose_code_label, epoch, steps, Nd, args)

        if epoch % args.save_freq == 0:
            # 各エポックで学習したモデルを保存
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            save_path_D = os.path.join(args.save_dir, 'epoch{}_D.pt'.format(epoch))
            torch.save(D_model, save_path_D)
            save_path_G = os.path.join(args.save_dir, 'epoch{}_G.pt'.format(epoch))
            torch.save(G_model, save_path_G)
            # 最後のエポックの学習前に生成した画像を１枚保存（学習の確認用）
            save_generated_image = convert_image(generated[0].cpu().data.numpy())
            save_path_image = os.path.join(args.save_dir, 'epoch{}_generatedimage.jpg'.format(epoch))
            misc.imsave(save_path_image, save_generated_image.astype(np.uint8))



def Learn_D(D_model, loss_criterion, loss_criterion_gan, optimizer_D, batch_image, generated, \
            batch_id_label, batch_pose_label, batch_ones_label, batch_zeros_label, epoch, steps, Nd, args):

    real_output = D_model(batch_image)
    syn_output = D_model(generated.detach())  # .detach() をすることで Generatorまでの逆伝播計算省略

    # id,真偽, pose それぞれのロスを計算
    L_id = loss_criterion(real_output[:, :Nd], batch_id_label)
    L_gan = loss_criterion_gan(real_output[:, Nd], batch_ones_label) + loss_criterion_gan(syn_output[:, Nd], batch_zeros_label)
    L_pose = loss_criterion(real_output[:, Nd+1:], batch_pose_label)

    d_loss = L_gan + L_id + L_pose

    d_loss.backward()
    optimizer_D.step()
    log_learning(epoch, steps, 'D', d_loss.data[0], args)

    # Discriminator の強さを判別
    flag_D_strong = Is_D_strong(real_output, syn_output, batch_id_label, batch_pose_label, Nd)

    return flag_D_strong


def Learn_G(D_model, loss_criterion, loss_criterion_gan, optimizer_G, generated, \
            batch_id_label, batch_ones_label, pose_code_label, epoch, steps, Nd, args):

    syn_output = D_model(generated)

    # id についての出力と元画像のラベル, 真偽, poseについての出力と生成時に与えたposeコード の ロスを計算
    L_id = loss_criterion(syn_output[:, :Nd], batch_id_label)
    L_gan = loss_criterion_gan(syn_output[:, Nd], batch_ones_label)
    L_pose = loss_criterion(syn_output[:, Nd+1:], pose_code_label)

    g_loss = L_gan + L_id + L_pose

    g_loss.backward()
    optimizer_G.step()
    log_learning(epoch, steps, 'G', g_loss.data[0], args)
