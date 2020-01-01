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

class CelebA_dataset(Dataset):
    def __init__(self, opt):
        print("Load file from :" ,opt.image_root)
        self.images = torch.load(opt.image_root)
        self.num_samples = len(self.images)

    def __getitem__(self, index):
        data = self.images[index]
        return data

    def __len__(self):
        return self.num_samples

def main():
    file_root = './hw3_data/face/train'
    csv_root = "./hw3_data/face/train.csv"
    train_dataset = ACGAN_Dataset(filepath = file_root ,csvpath = csv_root)
    train_loader = DataLoader(train_dataset,batch_size=8,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    print(len(train_loader.dataset))
    print(len(train_loader))
    for epoch in range(1):
        img,target = next(train_iter)

        im = plt.imshow(img[0])
        plt.show()
        print(img.shape)
        print(target[0])
        print(target[0,0,0,:])
# main()

if __name__ == '__main__':
    main()
