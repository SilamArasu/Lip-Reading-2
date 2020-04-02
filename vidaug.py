import random
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import numpy as np

def normalize(frames):
    """
    Changing the range of pixel intensity values to mean 0
    """
    # print("Executing normalize")
    frames = (frames - frames.min()) / frames.std()
    return frames

def vertical_flip(frames):
    """
    Vertically flip the video
    """
    # print("Executing vertical_flip")
    frames = np.flip(frames, 1)
    return frames

def pixelate(frames):
    """
    Downscale the image by the pixelation factor(75%) and then upscale it with nearest neighbour to original size
    """
    # print("Executing pixelate")
    for i in range(len(frames)):
        img = Image.fromarray(frames[i])
        imgSmall = img.resize((38,75),resample=Image.BILINEAR) # Original shape=(50,100). Resizing to 75%
        frames[i] = np.array(imgSmall.resize(img.size,Image.NEAREST))
    return np.array(frames)

def blur(frames):
    """
    Blur images using gaussian filter
    """
    # print("Executing blur")
    blurred = [gaussian_filter(x, sigma=1) for x in frames]
    return np.array(blurred)

def invert(frames):
    """
    Inverts the color of the video
    """
    # print("Executing invert")
    return np.invert(frames)

def add(frames):
    # print("Executing add")
    """
    Add a value to all pixel intesities in an video.
    """
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
    # print("Executing multiply")
    """
    Multiply all pixel intensities with given value.
    """
    data_final = []
    for i in range(len(frames)):
        image = frames[i].astype(np.float64)
        image *= 2
        image = np.where(image > 255, 255, image)
        image = np.where(image < 0, 0, image)
        image = image.astype(np.uint8)
        data_final.append(image.astype(np.uint8))
    return np.array(data_final)

def salt_pepper(frames):
    # print("Executing salt_pepper")
    """
    Salt - This sets a certain fraction of pixel intesities to 255 using noise, hence they become white.
    Pepper - This sets a certain fraction of pixel intesities to 255 using noise, hence they become black.
    """
    val = random.choice([0,255]) # Choose either black or white randomly
    data_final = []
    for i in range(len(frames)):
        img = frames[i].astype(np.float)
        noise = np.random.randint(50, size=img.shape).astype(np.float)
        img = np.where(noise == 0, val, img)
        data_final.append(img.astype(np.uint8))
    return np.array(data_final)

def select_augmentation(frames):
    # Randomly selects a augmentation to be applied
    func = random.choice([normalize, vertical_flip, pixelate, blur, invert, add, multiply, salt_pepper])
    return func(frames)