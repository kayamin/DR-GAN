import numpy as np
import pandas as pd
import scipy as sp
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MultiPIE_Dataset(Dataset):

    def __init__(self, image_attributes_df, transforms=None):

        # Create whole image path list

        self.whole_image_path = image_attributes_df['path'].values
        self.whole_image_Id = image_attributes_df['Id'].values
        self.whole_image_pose = image_attributes_df['pose'].values
        self.whole_image_illum = image_attributes_df['illum'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.whole_image_path)

    def __getitem__(self, idx):
        image = io.imread(self.whole_image_path[idx])
        Id = self.whole_image_Id[idx]
        pose = self.whole_image_pose[idx]
        illum = self.whole_image_illum[idx]

        if self.transforms:
            for t in self.transforms:
                image = t(image, pose)

        return [image, Id, pose, illum]

class FaceCrop(object):

    def __init__(self):
        slide_param = [[80, 0], [10, 20], [20,0], [0,0], [-40, -20], [-120, 0], [-90, 0], [-50,0], [10,0], [-10, -20], [50, -20], [40, -20], [70, 30]]
        yaw_order = [12, 0, 11, 10, 1, 2, 3, 9, 8, 4, 5, 7, 6]

        self.slide_param_ordered = slide_param_ordered = np.array(slide_param)[yaw_order]
        self.margin = 160

    def __call__(self, image, pose):

        def center_crop(image, margin, slide_x, slide_y):
            center_x = image.shape[1]//2 + 10
            left = center_x - margin + slide_x
            right = center_x + margin + slide_x

            center_y = image.shape[0]//2 - 40
            top = center_y + margin + slide_y
            bottom = center_y - margin + slide_y

            return image[bottom:top, left:right, :]

        return center_crop(image, self.margin, self.slide_param_ordered[pose,0], self.slide_param_ordered[pose,1])


class Resize(object):
    # assume image as H x W x C numpy uint8 [0, 255] array

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image, pose):
        resized_image = sp.misc.imresize(image, self.output_size)

        return resized_image


class RandomCrop(object):

    #  assume image  as H x W x C numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, image, pose):
        h, w = image.shape[:-1]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_image = image[top:top+new_h, left:left+new_w, :]

        return cropped_image


class RGBtoNormBGR(object):

    # convert RGB [0 255] H x W x C -> BGR [-1 1], C x H x W

    def __init__(self):

        return

    def __call__(self, image, pose):
        #[0,255] -> [-1,1]
        image = (image/255) *2 - 1
        # RGB -> BGR
        image = image[:,:,[2,1,0]]
        # H x W x C-> C x H x W
        image = image.transpose(2, 0, 1)

        return image
