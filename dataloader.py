import os
from os.path import join, exists
import glob
import numpy as np
import torch
import torch.utils.data as data
from util import cv2_imread

def augmentation(lr, hr):
    if np.random.random() < 0.5:
        lr = lr[:, ::-1, :, :]
        hr = hr[:, ::-1, :, :]
    if np.random.random() < 0.5:
        lr = lr[:, :, ::-1, :]
        hr = hr[:, :, ::-1, :]
    if np.random.random() < 0.5:
        lr = lr.transpose(0, 2, 1, 3)
        hr = hr.transpose(0, 2, 1, 3)

    return lr, hr

def double_crop(lr, hr, scale=4, size=32):
    B, H, W, C = lr.shape
    w0 = np.random.randint(0, W - size)
    h0 = np.random.randint(0, H - size)
    lr = lr[:, h0 : h0 + size, w0 : w0 + size,:]
    hr = hr[:, h0 * scale: (h0 + size) * scale, w0 * scale: (w0 + size) * scale, :]
    
    return augmentation(lr,hr)

def double_load(lr, hr, mode='train', crop_size=32, scale=4, num_frame=7):
    max_frame=len(lr)
    if mode == 'train':
        idx_st = np.random.randint(0, max_frame - num_frame + 1)
    elif mode == 'eval':
        idx_st = 15 - num_frame//2
    index = list(range(idx_st, idx_st + num_frame))
    
    lr_img=np.array([cv2_imread(lr[i]) for i in index])
    hr_img=np.array([cv2_imread(hr[i]) for i in index])
    if mode == 'train':
        lr_img, hr_img=double_crop(lr_img, hr_img, scale=scale, size=crop_size)

    return lr_img, hr_img

def single_aug(hr):
    if np.random.random() < 0.5:
        hr = hr[:, ::-1, :, :]
    if np.random.random() < 0.5:
        hr = hr[:, :, ::-1, :]
    if np.random.random() < 0.5:
        hr = hr.transpose(0, 2, 1, 3)

    return hr

def single_crop(hr, scale=4, size=32):
    B, H, W, C = hr.shape
    w0 = np.random.randint(0, W - size * scale + 1)
    h0 = np.random.randint(0, H - size * scale + 1)
    hr = hr[:, h0: h0 + size * scale, w0: w0 + size * scale, :]
    
    return single_aug(hr)

def single_load(hr, mode='train', crop_size=32,scale=4, num_frame=7):
    max_frame = len(hr)
    if mode == 'train':
        idx_st = np.random.randint(0, max_frame - num_frame + 1)
    elif mode == 'eval':
        idx_st = 15 - num_frame//2
    index = list(range(idx_st, idx_st + num_frame))

    hr_img = np.array([cv2_imread(hr[i]) for i in index])
    if mode == 'train':
        hr_img = single_crop(hr_img, size=crop_size)

    return hr_img

class loader(data.Dataset):
    def __init__(self, path, data_kind='single', mode='train', scale=4, crop_size=32, num_frame=7):
        if path.endswith('txt'):
            data_mode = 'txt'
            paths = open(path, 'rt').read().splitlines()
        else:
            data_mode = 'filefolder'
            paths = sorted(glob.glob(join(path, '*')))
            paths = [p for p in paths if os.path.isdir(p)]

        if data_kind == 'double':
            lqname = 'input{}'.format(scale)
            self.lqfiles=[sorted(glob.glob(join(p, lqname, '*'))) for p in paths]
        hqname = 'truth'
        self.hqfiles=[sorted(glob.glob(join(p, hqname, '*'))) for p in paths]

        self.data_kind = data_kind
        self.crop_size = crop_size
        self.scale = scale
        self.mode = mode
        self.num_frame = num_frame
        self.length = len(self.hqfiles)

        print('{} examples for {}'.format(self.length, mode))

    def __getitem__(self, index):
        if self.data_kind == 'double':
            LR, HR = double_load(self.lqfiles[index], self.hqfiles[index], mode=self.mode, crop_size=self.crop_size, 
                                scale=self.scale, num_frame=self.num_frame)
            LR = torch.from_numpy(LR / 255.).float().permute(3, 0, 1, 2)
        else:
            HR = single_load(self.hqfiles[index], mode=self.mode, crop_size=self.crop_size, 
                                scale=self.scale, num_frame=self.num_frame)
        HR = torch.from_numpy(HR / 255.).float().permute(3, 0, 1, 2)

        if self.data_kind == 'double':
            return LR, HR
        else:
            return HR

    def __len__(self):
        return self.length