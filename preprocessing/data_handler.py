# !/usr/bin/env python
# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from skimage import io, transform, color
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import cv2
import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

image_size = [224, 224, 3]
half_image_size = [int(i / 2) for i in image_size]


class Places(Dataset):
    def __init__(self,
                 image_dir, data_pattern="*.jpg", annotation_file_path=None,
                 transforms=None,
                 is_training=True, is_testing=False,
                 hard_example_mining_file_path=None, repeat_times=None):
        """
        step 0. get image file path list
        step 1. get annotation info and other info from csv file
        step 2. merge filepath_list and input_file_info, and clean merged_info to valid_input_info
        from step 0, 1, 2 ======> get the valid input info as self.valid_input_info

        step 3. get training or testing mode
        step 4. get transformation for training data

        :param image_dir:
        :param data_pattern:
        :param annotation_file_path:
        :param transforms:
        :param is_training:
        :param is_testing:
        """
        # step 1. get image file path list
        image_dir = image_dir[:-1] if image_dir[-1] == "/" else image_dir
        data_pattern = data_pattern[1:] if data_pattern[0] == "/" else data_pattern
        filepath_list = glob(image_dir + "/" + data_pattern)
        if len(filepath_list) == 0:
            raise ValueError("Can't find any file in the dir: {}".format(image_dir))

        # step 2. get annotation info and other info from csv file
        annotation_file_list = annotation_file_path.strip().split("/")
        annotation_file_path = "/".join(annotation_file_list[:-1]) + \
                               ("/train.txt" if is_training is True else "/test.txt")
        print annotation_file_path

        input_file_info = pd.read_csv(annotation_file_path, names=["filename"])

        input_file_info["filepath"] = input_file_info["filename"].map(
            lambda incomplete_filename: self._get_filepath_from_filename(filepath_list, str(incomplete_filename)))

        self.valid_input_info = input_file_info.dropna()

        if len(self.valid_input_info) == 0:
            raise ValueError("Empty Valid Info DataFrame.")

        # step 3. get transformation for training data
        if transforms is None:
            if is_testing or not is_training:
                self.transforms = T.Compose([
                    RGB2LAB(),
                    ToTensor()
                ])
            else:
                self.transforms = T.Compose([
                    RandomHorizontalFlip(),
                    RandomCrop([224, 224]),
                    RGB2LAB(),
                    ToTensor()
                ])

    @staticmethod
    def _get_filepath_from_filename(exist_filepath_list, incomplete_filename):
        """

        :param exist_filepath_list:
        :param incomplete_filename:
        :return:
        """

        for filepath in exist_filepath_list:
            if str(incomplete_filename) in filepath:
                image = np.asarray(Image.open(str(filepath)))

                if len(image.shape) == 3:
                    return str(filepath)

    def __len__(self):
        # print "Length of data: {}".format(self.valid_input_info.shape[0])
        return self.valid_input_info.shape[0]

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        one_input_info = self.valid_input_info.iloc[index]

        image_filepath = one_input_info["filepath"]

        real_image = Image.open(image_filepath)
        real_image.convert(mode="RGB")
        image = np.asarray(real_image)
        # print "Original Image:\n", image
        # plt.imshow(image)
        # plt.show()

        if len(image.shape) != 3:
            raise ValueError("Image Error: {}".format(image_filepath))

        if self.transforms:
            image = self.transforms(image)

        # np_image = image.numpy().transpose(1, 2, 0)
        # l = np.expand_dims(np_image[:, :, 0] * 100.0, axis=2)
        # ab = np_image[:, :, 1:3] * 200.0 - 100.0
        # np_image = np.concatenate([l, ab], axis=2)
        # np_image = color.lab2rgb(np_image)
        # plt.imshow(np_image)
        # plt.show()

        input_image = torch.unsqueeze(image[0, :, :], 0)
        target_image = image[1:, :, :]
        min_value = torch.min(image)
        max_value = torch.max(image)

        return input_image, target_image, one_input_info["filename"], min_value, max_value


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, list))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        new_image = transform.resize(image, (new_h, new_w, 3), mode='constant')

        # print "after flip:", new_image.shape

        return new_image


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, min_ratio_xyz=None):
        assert isinstance(output_size, (int, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        if not min_ratio_xyz:
            self.min_ratio_xyz = [0.875,  0.875]
        else:
            self.min_ratio_xyz = min_ratio_xyz

    def __call__(self, sample):
        image = sample

        image_shape = image.shape
        new_image_size = np.random.randint(int(self.min_ratio_xyz[0] * image_shape[0]), image_shape[0])
        new_min_x = np.random.randint(0, image_shape[0] - new_image_size)
        new_min_y = np.random.randint(0, image_shape[1] - new_image_size)

        cropped_image = image[new_min_x:new_min_x + new_image_size,
                        new_min_y:new_min_y + new_image_size]

        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        new_image = transform.resize(cropped_image, (new_h, new_w, 3), mode='constant')

        # print "after crop:", new_image.shape

        return new_image


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""
    def __init__(self):
        pass

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image = sample

        if np.random.uniform(0, 1) > 0.5:
            image = np.flip(image, 0).copy()

        if np.random.uniform(0, 1) > 0.5:
            image = np.flip(image, 1).copy()

        # print "after flip:", image.shape

        return image


class RGB2LAB(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # print image
        # image = image.astype(np.float32)
        # image = image / 255.0

        lab_image = color.rgb2lab(image)

        l = np.expand_dims(lab_image[:, :, 0] / 100, axis=2)
        ab = (lab_image[:, :, 1:3] + 100.) / 200.
        lab_image = np.concatenate([l, ab], axis=2)

        # print "L", np.min(lab_image[:, :, 0]), np.max(lab_image[:, :, 0])
        # print "ab", np.min(lab_image[:, :, 1:3]), np.max(lab_image[:, :, 1:3])
        # print "after lab:", image.shape

        return lab_image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image)


def main():
    # test load dataset with dataloader
    places_dataset = Places("../data/testSetPlaces205_resize/testSet_resize",
                                     "*.jpg",
                                     "../data/testSetPlaces205_resize/imgList.csv",
                                 is_training=False,
                                 is_testing=True)
    places_dataloader = DataLoader(places_dataset, batch_size=2, shuffle=False, num_workers=4)

    # test directly load dataset
    start_time = time.time()
    for i in range(len(places_dataset)):
        image, label, filename, _, _ = places_dataset[i]
        print image.shape
        print label.shape
        print filename

        if i == 1:
            break
    end_time = time.time()
    print "Time is: {}".format(end_time - start_time)

    # test 4
    start_time = time.time()
    for i, (image, label, filename, _, _) in enumerate(places_dataloader):
        # print image[0][0].numpy()
        # plt.imshow(image[0][0].numpy() / np.max(image[0][0].numpy()))
        # plt.show()
        #
        # x = label[0][0].numpy()
        # plt.imshow(x)
        # plt.show()
        #
        # x = label[0][1].numpy()
        # plt.imshow(x)
        # plt.show()

        if i == 2:
            break
    end_time = time.time()
    print "Time is {}".format(end_time - start_time)


if __name__ == '__main__':
    main()
