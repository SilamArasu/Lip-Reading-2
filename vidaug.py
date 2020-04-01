import random
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import numpy as np

def generalize(frames):
    frames = (frames - frames.min(axis=0)) / frames.std(axis=0)
    return frames

def vertical_flip(frames):
    frames = np.flip(frames, 1)
    return frames

def pixelate(frames):
    for i in range(len(frames)):
        img = Image.fromarray(frames[i])
        imgSmall = img.resize((38,75),resample=Image.BILINEAR) # Original shape=(50,100). Resizing to 75%
        frames[i] = np.array(imgSmall.resize(img.size,Image.NEAREST))
    return np.array(frames)

def blur(frames):
    blurred = [gaussian_filter(x, sigma=1) for x in frames]
    return np.array(blurred)

def invert(frames):
    return np.invert(frames)

def add(frames):
    data_final = []
    for i in range(len(frames)):
        image = frames[i].astype(np.int32)
        image += 100    # Add 100 to each pixel
        image = np.where(image > 255, 255, image)
        image = np.where(image < 0, 0, image)
        image = image.astype(np.uint8)
        data_final.append(image.astype(np.uint8))
    return np.array(data_final)

def multiply(frames):
    data_final = []
    for i in range(len(frames)):
        image = frames[i].astype(np.float64)
        image *= 2
        image = np.where(image > 255, 255, image)
        image = np.where(image < 0, 0, image)
        image = image.astype(np.uint8)
        data_final.append(image.astype(np.uint8))
    return np.array(data_final)

def select_augmentation(frames):
    func = random.choice([generalize, vertical_flip, pixelate])
    return func(frames)