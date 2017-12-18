#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn, optim
from torch.autograd import Variable
import pdb


class Discriminator(nn.Module):
    """
    multi-task CNN for identity and pose classification

    ### init
    Nd : Number of identitiy to classify
    Np : Number of pose to classify

    """

    def __init__(self, Nd, Np, channel_num):
        super(Discriminator, self).__init__()
        convLayers = create_convs(channel_num)

        self.convLayers = nn.Sequential(*convLayers)
        self.fc = nn.Linear(320, Nd+1+Np)

        # 重みは全て N(0, 0.02) で初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, input):
        # 畳み込み -> 平均プーリングの結果 B x 320 x 1 x 1の出力を得る
        x = self.convLayers(input)
        x = x.view(-1, 320)

        # 全結合
        x = self.fc(x)  # Bx320 -> B x (Nd+1+Np)

        return x


class Generator(nn.Module):
    """
    Encoder/Decoder conditional GAN conditioned with pose vector and noise vector

    ### init
    Np : Dimension of pose vector (Corresponds to number of dicrete pose classes of the data)
    Nz : Dimension of noise vector

    """

    def __init__(self, Np, Nz, channel_num, multi=False, images_perID=None):
        super(Generator, self).__init__()
        self.features = []
        self.multi_flag = multi
        self.Nf = 320
        if self.multi_flag:
            self.Nf += 1
            self.images_perID = images_perID

        G_enc_convLayers = create_convs(channel_num, self.Nf)
        self.G_enc_convLayers = nn.Sequential(*G_enc_convLayers)

        G_dec_convLayers = [
            nn.ConvTranspose2d(320, 160, 3, 1, 1, bias=False),  # Bx320x6x6 -> Bx160x6x6
            nn.BatchNorm2d(160),
            nn.ELU(),
            nn.ConvTranspose2d(160, 256, 3, 1, 1, bias=False),  # Bx160x6x6 -> Bx256x6x6
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.ConvTranspose2d(256, 256, 3, 2, 0, bias=False),  # Bx256x6x6 -> Bx256x13x13
            nn.BatchNorm2d(256),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(256, 128, 3, 1, 1, bias=False),  # Bx256x12x12 -> Bx128x12x12
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 192, 3, 1, 1, bias=False),  # Bx128x12x12 -> Bx192x12x12
            nn.BatchNorm2d(192),
            nn.ELU(),
            nn.ConvTranspose2d(192, 192, 3, 2, 0, bias=False),  # Bx128x12x12 -> Bx192x25x25
            nn.BatchNorm2d(192),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(192, 96, 3, 1, 1, bias=False),  # Bx192x24x24 -> Bx96x24x24
            nn.BatchNorm2d(96),
            nn.ELU(),
            nn.ConvTranspose2d(96, 128, 3, 1, 1, bias=False),  # Bx96x24x24 -> Bx128x24x24
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.ConvTranspose2d(128, 128, 3, 2, 0, bias=False),  # Bx128x24x24 -> Bx128x49x49
            nn.BatchNorm2d(128),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(128, 64, 3, 1, 1, bias=False),  # Bx128x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64,  3, 1, 1, bias=False),  # Bx64x48x48 -> Bx64x48x48
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 3, 2, 0, bias=False),  # Bx64x48x48 -> Bx64x97x97
            nn.BatchNorm2d(64),
            nn.ELU(),
            Crop([0, 1, 0, 1]),
            nn.ConvTranspose2d(64, 32, 3, 1, 1, bias=False),  # Bx64x96x96 -> Bx32x96x96
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.ConvTranspose2d(32, channel_num, 3, 1, 1, bias=False),  # Bx32x96x96 -> Bxchx96x96
            nn.Tanh()
        ]

        self.G_dec_convLayers = nn.Sequential(*G_dec_convLayers)

        self.G_dec_fc = nn.Linear(320+Np+Nz, 320*6*6)

        # 重みは全て N(0, 0.02) で初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or \
               isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)

    def forward(self, input, pose, noise):

        x = self.G_enc_convLayers(input)  # Bxchx96x96 -> Bx320x1x1

        x = x.view(-1, self.Nf)

        if self.multi_flag and self.images_perID:
            x = weight_sum_feature(x, self.images_perID)

        x = torch.cat([x, pose, noise], 1)  # Bx320 -> B x (320+Np+Nz)
        x = self.G_dec_fc(x)  # B x (320+Np+Nz) -> B x (320x6x6)
        x = x.view(-1, 320, 6, 6)  # B x (320x6x6) -> B x 320 x 6 x 6
        x = self.G_dec_convLayers(x)  # B x 320 x 6 x 6 -> Bxchx96x96

        return x


# nn.Module を継承しても， super でコンストラクタを呼び出さないと メンバ変数 self._modues が
# 定義されずに後の重み初期化の際にエラーを出す
# sef._modules はモジュールが格納するモジュール名を格納しておくリスト

class Crop(nn.Module):
    """
    Generator でのアップサンプリング時に， ダウンサンプル時のZeroPad2d と逆の事をするための関数
    論文著者が Tensorflow で padding='SAME' オプションで自動的にパディングしているのを
    ダウンサンプル時にはZeroPad2dで，アップサンプリング時には Crop で実現

    ### init
    crop_list : データの上下左右をそれぞれどれくらい削るか指定
    """

    def __init__(self, crop_list):
        super(Crop, self).__init__()

        # crop_lsit = [crop_top, crop_bottom, crop_left, crop_right]
        self.crop_list = crop_list

    def forward(self, x):
        B, C, H, W = x.size()
        x = x[:, :, self.crop_list[0]:H - self.crop_list[1], self.crop_list[2]:W - self.crop_list[3]]

        return x


def weight_sum_feature(x, n):
    """
    重みの出力の部分にだけシグモイド関数をかけ， その重みを用いて n枚の画像の出力結果を足し合わせる
    入力： nBx321x1x1 -> 出力: B x 320x1x1

    n : 一人にあたり何枚の画像をデータとして渡しているのか
    B : バッチ毎に何人分(１人n枚)の画像をデータとして渡しているのか

    """
    # nBx320x1x1 -> Bx320x1x1

    B = x.size(0)//n
    weight = x[:, -1].unsqueeze(1).sigmoid()
    features = x*weight
    features = features[:, :-1].split(n, 0)
    features = torch.cat(features, 1)
    features_summed = features.sum(0, keepdim=True)
    features_summed = features_summed.view(B, -1)

    return features_summed


def create_convs(channel_num, n_last=320):
    return [
        nn.Conv2d(channel_num, 32, 3, 1, 1, bias=False),  # Bxchx96x96 -> Bx32x96x96
        nn.BatchNorm2d(32),
        nn.ELU(),
        nn.Conv2d(32, 64, 3, 1, 1, bias=False),  # Bx32x96x96 -> Bx64x96x96
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx64x96x96 -> Bx64x97x97
        nn.Conv2d(64, 64, 3, 2, 0, bias=False),  # Bx64x97x97 -> Bx64x48x48
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # Bx64x48x48 -> Bx64x48x48
        nn.BatchNorm2d(64),
        nn.ELU(),
        nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # Bx64x48x48 -> Bx128x48x48
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.ZeroPad2d((0, 1, 0, 1)),  # Bx128x48x48 -> Bx128x49x49
        nn.Conv2d(128, 128, 3, 2, 0, bias=False),  # Bx128x49x49 -> Bx128x24x24
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.Conv2d(128, 96, 3, 1, 1, bias=False),  # Bx128x24x24 -> Bx96x24x24
        nn.BatchNorm2d(96),
        nn.ELU(),
        nn.Conv2d(96, 192, 3, 1, 1, bias=False),  # Bx96x24x24 -> Bx192x24x24
        nn.BatchNorm2d(192),
        nn.ELU(),
        nn.ZeroPad2d((0, 1, 0, 1)),                      # Bx192x24x24 -> Bx192x25x25
        nn.Conv2d(192, 192, 3, 2, 0, bias=False),  # Bx192x25x25 -> Bx192x12x12
        nn.BatchNorm2d(192),
        nn.ELU(),
        nn.Conv2d(192, 128, 3, 1, 1, bias=False),  # Bx192x12x12 -> Bx128x12x12
        nn.BatchNorm2d(128),
        nn.ELU(),
        nn.Conv2d(128, 256, 3, 1, 1, bias=False),  # Bx128x12x12 -> Bx256x12x12
        nn.BatchNorm2d(256),
        nn.ELU(),
        nn.ZeroPad2d((0, 1, 0, 1)),  # Bx256x12x12 -> Bx256x13x13
        nn.Conv2d(256, 256, 3, 2, 0, bias=False),  # Bx256x13x13 -> Bx256x6x6
        nn.BatchNorm2d(256),
        nn.ELU(),
        nn.Conv2d(256, 160, 3, 1, 1, bias=False),  # Bx256x6x6 -> Bx160x6x6
        nn.BatchNorm2d(160),
        nn.ELU(),
        nn.Conv2d(160, n_last, 3, 1, 1, bias=False),  # Bx160x6x6 -> Bx320x6x6
        nn.BatchNorm2d(n_last),
        nn.ELU(),
        nn.AvgPool2d(6, stride=1),  # Bx320x6x6 -> Bx320x1x1
    ]
