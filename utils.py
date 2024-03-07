import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt


def noise(sigma, img):
    return img + np.random.normal(0, sigma / 255., img.shape)

files = os.listdir('data/train')

sigma = 25

for file in files:
    img = plt.imread(os.path.join('data/train', file))
    img_noise = noise(sigma, img)
    plt.imsave(os.path.join('data/train_noise', file), img_noise, cmap='gray')

