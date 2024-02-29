
import torch
import torch.nn as nn
from torchvision.utils import save_image


import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as tr

import polip as pl
#mine library for faster coding

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse



from discriminator import Discriminator
from generator import Generator
from utils import *
from ds import MapDataset

from tqdm import tqdm



device = pl.decider("mps")



parser = argparse.ArgumentParser(description = "Train a pix2pix")
parser.add_argument("--train_dir", type = str, default = "data/maps/maps/train")
parser.add_argument("--val_dir", type = str, default = "data/maps/maps/val")

args = parser.parse_args()


ds = MapDataset(
    root_dir = args.train_dir,
)

dl = torch.utils.data.DataLoader(ds,
                                 batch_size = 4, shuffle = False,
                                 pin_memory = True)

val_ds = MapDataset(
    root_dir = args.val_dir,
)

val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size = 4, shuffle = True, pin_memory = True
)





gen = Generator().to(device)
dis = Discriminator().to(device)


opt_disc = torch.optim.Adam(dis.parameters(),
                            lr = 2e-4, betas = (0.5, 0.999) )
opt_gen = torch.optim.Adam(gen.parameters(),
                           lr = 2e-4, betas = (0.5, 0.999))

bce_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5
        os.makedirs(folder, exist_ok=True)
        save_image(y_fake, os.path.join(folder, f"y_gen_{epoch}.png"))
        save_image(x * 0.5 + 0.5, os.path.join(folder, f"input_{epoch}.png"))
        if epoch == 1:
            save_image(y * 0.5 + 0.5, os.path.join(folder, f"label_{epoch}.png"))
    gen.train()




num_epochs = 500
for epoch in range(num_epochs):
    loop = tqdm(dl, leave=True, total=len(dl))

    for idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type="cuda"):
            y_fake = gen(x)
            D_real = dis(x, y)
            D_real_loss = bce_loss(D_real, torch.ones_like(D_real))
            D_fake = dis(x, y_fake.detach())
            D_fake_loss = bce_loss(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        dis.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.autocast(device_type="cuda"):
            D_fake = dis(x, y_fake)
            G_fake_loss = bce_loss(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * 100
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
        torch.save(gen.state_dict(), f"weights/generator_epoch_{epoch}.pth")
        torch.save(dis.state_dict(), f"weights/discriminator_epoch_{epoch}.pth")


    save_some_examples(gen, val_dl, epoch, "predictions")