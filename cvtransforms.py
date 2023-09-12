# coding: utf-8
import random
import cv2
import numpy as np


class Resize(object):
    "resized the given numpy array to static size"

    def __init__(self, size):
        self.size = size

    def __call__(self, batch_img):
        return cv2.resize(
            batch_img, self.size, interpolation=cv2.INTER_LANCZOS4)


def CenterCrop_f(batch_img, size):
    #print('the batch_img shape is: ',batch_img.shape)
    w, h = batch_img.shape[-1], batch_img.shape[-2]
    th, tw = size
    x1 = int(round((w - tw))/2.)
    y1 = int(round((h - th))/2.)
    img = batch_img[:,:,y1:y1+th, x1:x1+tw]
    return img


class CenterCrop(object):
    "centercrop numpy array to static size"

    def __init__(self, size):
        self.size = size

    def __call__(self, batch_img):
        return CenterCrop_f(
            batch_img, self.size)


def RandomCrop_f(batch_img, size):
    #print('the batch_img shape is: ',batch_img.shape)
    w, h = batch_img.shape[-1], batch_img.shape[-2]
    tw ,th = size
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    img = batch_img[:,:,y1:y1+th, x1:x1+tw]
    return img


class RandomCrop(object):
    "randomcrop the given numpy array to static size"

    def __init__(self, size):
        self.size = size

    def __call__(self, batch_img):
        return RandomCrop_f(
            batch_img, self.size)


class HorizontalFlip(object):
    "flip the given image horizontally"

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, batch_img):
        if random.random() > self.p:
            batch_img = batch_img[:,:,:,::-1]
        return batch_img

class RGB2GREY(object):
    "convert RGB to GREY image"
    def __init__(self):
        pass

    def __call__(self, batch_img):
        batch_img = cv2.cvtColor(batch_img, cv2.COLOR_BGR2GRAY)[:,:,:,:,None]
        return batch_img
        
def RandomDrop(batch_img):
    i = 0
    for j in range(batch_img.shape[0]):
        if 0.01 < random.random() or min(10, 0.2*batch_img.shape[0]) < 1.*(j - i):
            batch_img[i] = batch_img[j]
            i += 1
    for j in range(i, batch_img.shape[0]):
        batch_img[j] = batch_img[j - 1]
    return batch_img


def ColorNormalize(batch_img):
    mean = 0.413621
    std = 0.1700239
    batch_img = (batch_img - mean) / std
    return batch_img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, scale=None):
        for t in self.transforms:
            img = t(img)
        return img
