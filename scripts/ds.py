import os


import torch

from utils import *
import torchvision.transforms as tr

import numpy as np
from PIL import Image


transform_b = A.Compose([
    A.Resize(256, 256)
])

transform_l = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.ColorJitter(p = 0.2),
    A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ToTensorV2()
])

transform_r = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ToTensorV2()
])


class MapDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600]
        target_image = image[:, 600:]

        augmentations = transform_b(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = transform_l(image=input_image)["image"]
        target_image = transform_r(image=target_image)["image"]

        return input_image, target_image
