import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import os.path
import sys
import string
import pandas as pd
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

def preprocessor(path, image_size):
    figsize = image_size
    images = []
    file_list = os.listdir(path)
    file_list.sort()
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        # transforms.CenterCrop(opt.imageSize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for i, file in enumerate(tqdm(file_list)):
        img = Image.open(os.path.join(path, file))
        img = transform(img)
        images.append(img)

    images = torch.stack(images)
    images = images.transpose(1, 2).transpose(2, 3)
    return images



def main():
    file_root = '../dataset/celeba/train/images'
    save_root = '../dataset/celeba/train/preprocessing'
    image_size = 64
    tensor = preprocessor(file_root, image_size)
    torch.save(tensor, os.path.join(save_root,'celebA_64.pt'))
# main()

if __name__ == '__main__':
    main()
