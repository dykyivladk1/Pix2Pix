from torchvision.utils import save_image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import cv2

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
    A.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),
    ToTensorV2()
])

def save_some_examples(gen, image_sample, epoch, folder):
    device = torch.device("mps")

    image = cv2.imread(image_sample)
    image = transform_b(image)
    image = image.to(device)

    gen.eval()
    with torch.no_grad():
        y_fake = gen(image.unsqueeze(0))
        y_fake = y_fake * 0.5 + 0.5
        save_image(y_fake, f"{folder}/y_gen_{epoch}.png")
        save_image(image * 0.5 + 0.5, f"{folder}/input_{epoch}.png")
    gen.train()
