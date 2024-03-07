import os
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import sys


def noise(sigma, img):
    return img + np.random.normal(0, sigma / 255., img.shape)

def create_patches(img, patch_size, stride):
    patches = []
    for i in range(0, img.shape[0]-patch_size, stride):
        for j in range(0, img.shape[1]-patch_size, stride):
            patch = img[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return patches

def augment(img):
    val = random.randint(1, 8)
    if val == 1:
        pass
    elif val == 2:
        img =  np.flipud(img)
    elif val == 3:
         out = np.rot90(img)
    elif val == 4:
        img =  np.flipud(np.rot90(img))
    elif val == 5:
        img =  np.rot90(img, k=2)
    elif val == 6:
        img =  np.flipud(np.rot90(img, k=2))
    elif val == 7:
        img = np.rot90(img, k=3)
    elif val == 8:
        img = np.flipud(np.rot90(img, k=3))
    
    return img


if __name__ == '__main__':
    train_files = os.listdir('data/train')

    sigma = 25
    patch = 40
    stride = 10 # paper stride = 1

    # 196 patches per image, 10 images

    for file in train_files:
        img = plt.imread(os.path.join('data/train', file))
        img_noise = noise(sigma, img)
        plt.imsave(os.path.join('data/train_noise', file), img_noise, cmap='gray')

        patches_aug_noisy = [augment(patch) for patch in create_patches(img_noise, patch, stride)]
        patches_aug = [augment(patch) for patch in create_patches(img, patch, stride)]

    test_files = os.listdir('data/test')
    for file in test_files:
        img = plt.imread(os.path.join('data/test', file))
        img_noise = noise(sigma, img)
        plt.imsave(os.path.join('data/test_noise', file), img_noise, cmap='gray')



