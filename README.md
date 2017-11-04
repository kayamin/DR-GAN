# [Disentangled Representation Learning GAN for Pose-Invariant Face Recognition](http://cvlab.cse.msu.edu/project-dr-gan.html)
# * Still under Development and haven't check the result of this implimentation (2017/9/25)

- authors: Luan Tran, Xi Yin, Xiaoming Liu
- CVPR2017: http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf
- Pytorch implimentation of DR-GAN (updated version in "Representation Learning by Rotating Your Faces")
- Powered by [DL HACKS](http://deeplearning.jp/hacks/)

## Requirements
- python 3.6
- pytorch 0.2.0
- numpy 1.13.1
- scipy 0.18.1
- matplotlib 2.0.0

## How to use

### Single-Image DR-GAN
1. modify DataLoader function at main.py to define dataloader which is applicable to your data
    - data needs to have ID and pose lables corresponds to each image
    - if you don't have, "-random" option allow you to see how the code works with meanless random data.
    > python main.py -random

2. Run main.py to train models
      - trained models and Loss_log will be saved at "DR_GAN/snapshot/Single" by default
      > python main.py -random  

3. Generate Image with arbitrary pose
      - use "-generate" option
      - specify leaned model by "-snapshot" option
      - generated images will be saved at specified sanpshot directory
      > python main.py -random -generate -snapshot=snapshot/Single/2017-09-22_20-31-08/epoch1


### Multi-Image DR-GAN
1. modify DataLoader function at main.py to define dataloader which is applicable to your data
      - data needs to have ID and pose lables corresponds to each image
      - if you don't have, "-random" option allow you to see how the code works with meanless random data.
      > python main.py -multi-DRGAN -images-perID=4 -random

2. Run main.py with "-multi-DRGAN" and "-images-perID" option
      - Multi-Image DR-GAN assumes input data to have *N* images per person and in my code, they should be sequentially aligned. So change *N* depends on your data.
      - input data size have to be divisible by batch size
      - batch size have to be divisible by images_perID
      - trained modles and Loss_log will be saved at "DR_GAN/snapshot/Multi" by default
      > python main.py -multi-DRGAN -images-perID=4 -random

3. Generate Image with arbitrary pose
      - use "-generate" option
      - specify leaned model by "-snapshot" option
      - generated images will be saved at specified sanpshot directory
      > python main.py -random -multi-DRGAN -generate -images-perID=4 -snapshot=snapshot/Multi/2017-09-22_23-03-50/epoch5
