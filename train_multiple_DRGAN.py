#!/usr/bin/env python
# encoding: utf-8

import os
import numpy as np
import pdb
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.autograd import Variable
from util.Is_D_strong import Is_D_strong


def train_multiple_DRGAN(images, id_labels, pose_labels, Nd, Np, Nz, D_model, G_model, args):
    if args.cuda:
        D_model.cuda()
        G_model.cuda()

    D_model.train()
    G_model.train()

    lr_Adam = args.lr
    # m_Adam = 0.5

    image_size = images.shape[0]
    epoch_time = np.ceil(image_size / args.batch_size).astype(int)

    optimizer_D = optim.Adam(D_model.parameters(), lr = lr_Adam)
    optimizer_G = optim.Adam(G_model.parameters(), lr = lr_Adam)
    loss_criterion = nn.CrossEntropyLoss()

    loss_log = []
    steps = 0

    flag_D_strong  = False
    for epoch in range(1,args.epochs+1):
        for i in range(epoch_time):
            D_model.zero_grad()
            G_model.zero_grad()
            start = i*args.batch_size
            end = start + args.batch_size
            batch_image = torch.FloatTensor(images[start:end])
            batch_id_label = torch.LongTensor(id_labels[start:end])
            batch_id_label_unique = torch.LongTensor(batch_id_label[::args.images_perID])

            batch_pose_label = torch.LongTensor(pose_labels[start:end])
            minibatch_size_unique = len(batch_image) // args.images_perID
            syn_id_label = torch.LongTensor(Nd*np.ones(minibatch_size_unique).astype(int))

            # ノイズと姿勢コードを生成
            fixed_noise = torch.FloatTensor(np.random.uniform(-1,1, (minibatch_size_unique, Nz)))
            pose_code = np.zeros((minibatch_size_unique, Np))
            tmp  = np.random.randint(Np, size=minibatch_size_unique)
            pose_code[:, tmp] = 1
            pose_code = torch.FloatTensor(pose_code) # Condition 付に使用
            pose_code_label = torch.LongTensor(tmp) # CrossEntropy 誤差に使用


            if args.cuda:
                batch_image, batch_id_label, batch_id_label_unique, batch_pose_label, syn_id_label = \
                    batch_image.cuda(), batch_id_label.cuda(), batch_id_label_unique.cuda(), batch_pose_label.cuda(), syn_id_label.cuda()

                fixed_noise, pose_code, pose_code_label = \
                    fixed_noise.cuda(), pose_code.cuda(), pose_code_label.cuda()

            batch_image, batch_id_label, batch_id_label_unique, batch_pose_label, syn_id_label = \
                Variable(batch_image), Variable(batch_id_label), Variable(batch_id_label_unique), Variable(batch_pose_label), Variable(syn_id_label)

            fixed_noise, pose_code, pose_code_label = \
                Variable(fixed_noise), Variable(pose_code), Variable(pose_code_label)

            # Generatorでイメージ生成
            generated = G_model(batch_image, pose_code, fixed_noise)

            steps += 1

            # バッチ毎に交互に D と G の学習，　Dが90%以上の精度の場合は 1:4の比率で学習
            if flag_D_strong:

                if i%5 == 0:
                    # Discriminator の学習
                    real_output = D_model(batch_image)
                    syn_output = D_model(generated.detach()) # .detach() をすることでGeneratorのパラメータを更新しない

                    # id についての出力とラベル, pose についての出力とラベル それぞれの交差エントロピー誤差を計算
                    d_loss = loss_criterion(real_output[:, :Nd+1], batch_id_label) +\
                                            loss_criterion(real_output[:, Nd+1:], batch_pose_label) +\
                                            loss_criterion(syn_output[:, :Nd+1], syn_id_label)

                    d_loss.backward()
                    optimizer_D.step()
                    print("EPOCH : {0}, step : {1}, D : {2}".format(epoch, steps, d_loss.data[0]))

                    # Discriminator の強さを判別
                    flag_D_strong = Is_D_strong(real_output, syn_output, batch_id_label, batch_pose_label, syn_id_label, Nd)

                else:
                    # Generatorの学習
                    syn_output=D_model(generated)

                    # id についての出力と元画像のラベル, poseについての出力と生成時に与えたposeコード それぞれの交差エントロピー誤差を計算
                    g_loss = loss_criterion(syn_output[:, :Nd+1], batch_id_label_unique) +\
                        loss_criterion(syn_output[:, Nd+1:], pose_code_label)

                    optimizer_G.step()
                    print("EPOCH : {0}, step : {1}, G : {2}".format(epoch, steps, g_loss.data[0]))

            else:

                if i%2==0:
                    # Discriminator の学習
                    real_output = D_model(batch_image)
                    syn_output = D_model(generated.detach()) # .detach() をすることでGeneratorのパラメータを更新しない

                    # id についての出力とラベル, pose についての出力とラベル それぞれの交差エントロピー誤差を計算
                    d_loss = loss_criterion(real_output[:, :Nd+1], batch_id_label) +\
                                            loss_criterion(real_output[:, Nd+1:], batch_pose_label) +\
                                            loss_criterion(syn_output[:, :Nd+1], syn_id_label)

                    d_loss.backward()
                    optimizer_D.step()
                    print("EPOCH : {0}, step : {1}, D : {2}".format(epoch, steps, d_loss.data[0]))

                    # Discriminator の強さを判別
                    flag_D_strong = Is_D_strong(real_output, syn_output, batch_id_label, batch_pose_label, syn_id_label, Nd)

                else:
                    # Generatorの学習
                    syn_output=D_model(generated)

                    # id についての出力と元画像のラベル, poseについての出力と生成時に与えたposeコード それぞれの交差エントロピー誤差を計算
                    g_loss = loss_criterion(syn_output[:, :Nd+1], batch_id_label_unique) +\
                        loss_criterion(syn_output[:, Nd+1:], pose_code_label)

                    optimizer_G.step()
                    print("EPOCH : {0}, step : {1}, G : {2}".format(epoch, steps, g_loss.data[0]))

        # エポック毎にロスの保存
        loss_log.append([epoch, d_loss.data[0], g_loss.data[0]])
        # 各エポックで学習したモデルを保存
        if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
        save_path_D = os.path.join(args.save_dir,'epoch{}_D.pt'.format(epoch))
        torch.save(D_model, save_path_D)
        save_path_G = os.path.join(args.save_dir,'epoch{}_G.pt'.format(epoch))
        torch.save(G_model, save_path_G)

    # 学習終了後に，全エポックでのロスの変化を画像として保存
    loss_log = np.array(loss_log)
    plt.plot(loss_log[:,1], label="Discriminative Loss")
    plt.plot(loss_log[:,2], label="Generative Loss")
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    filename = os.path.join(args.save_dir, 'Loss_log.png')
    plt.savefig(filename, bbox_inches='tight')
