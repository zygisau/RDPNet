import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, samples):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1, img2, mask = samples
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1))
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1))
        mask = np.expand_dims(np.array(mask).astype(np.float32), axis=0)

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        mask = torch.from_numpy(mask).float()

        return img1, img2, mask


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, samples):
        img1, img2, mask = samples
        img1 = np.array(img1).astype(np.float32)
        img2 = np.array(img2).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img1 /= 255.0
        img1 -= self.mean
        img1 /= self.std
        img2 /= 255.0
        img2 -= self.mean
        img2 /= self.std
        return img1, img2, mask


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, samples, is_mask=False):
        img1, img2, mask = samples

        assert img1.size == mask.size and img2.size == mask.size

        img1 = img1.resize(self.size, Image.BILINEAR)
        img2 = img2.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return img1, img2, mask


train_transforms = transforms.Compose([
    # RandomHorizontalFlip(),
    # RandomVerticalFlip(),
    # RandomFixRotate(),
    # RandomRotate(30),
    FixedResize(256),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensor()])

test_transforms = transforms.Compose([
    FixedResize(256),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensor()])
