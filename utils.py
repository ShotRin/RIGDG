"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

import numpy as np
from PIL import Image
import random
import torch


def save_image_array(img_array, fname):
    """
    img_array: [H, W, C], values in [-1, 1]
    fname: save path
    """
    if img_array.shape[2] == 1:
        img_array = img_array[:, :, 0]  # grayscale

    img_array = (img_array * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_array).save(fname)


def save_image_array1(img_array, fname):
    channels = img_array.shape[2]
    resolution = img_array.shape[-1]
    img_rows = img_array.shape[0]
    img_cols = img_array.shape[1]

    # 이미지 배열을 채널 우선 (C, H, W) 형식으로 구성합니다.
    img = np.full([channels, resolution * img_rows, resolution * img_cols], 0.0)
    for r in range(img_rows):
        for c in range(img_cols):
            img[:,
                (resolution * r): (resolution * (r + 1)),
                (resolution * (c % 10)): (resolution * ((c % 10) + 1))
               ] = img_array[r, c]

    img = (img * 127.5 + 127.5).astype(np.uint8)
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.rollaxis(img, 0, 3)

    Image.fromarray(img).save(fname)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import matplotlib.pyplot as plt
import os

def save_latent_histogram(z_cond, epoch):
    z_np = z_cond.detach().cpu().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(z_np, bins=100)
    plt.title(f"z_cond Distribution @ Epoch {epoch}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    os.makedirs("histograms", exist_ok=True)
    plt.savefig(f"histograms/z_cond_epoch_{epoch}.png")
    plt.close()  # 메모리 누수 방지