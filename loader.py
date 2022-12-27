import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class CDDLoader(Dataset):
    def __init__(self, annotations_files, img_dirs, transform=None, append_filename=False):
        self.img_a = pd.read_csv(annotations_files[0])
        self.img_b = pd.read_csv(annotations_files[1])
        self.img_OUT = pd.read_csv(annotations_files[2])
        self.img_dirs = img_dirs
        self.transform = transform
        self.append_filename = append_filename

    def __len__(self):
        return len(self.img_a)

    def __getitem__(self, idx):
        img_a_path = os.path.join(self.img_dirs[0], self.img_a.iloc[idx, 0])
        img_b_path = os.path.join(self.img_dirs[1], self.img_b.iloc[idx, 0])
        img_out_path = os.path.join(self.img_dirs[2], self.img_OUT.iloc[idx, 0])
        image_a = Image.open(img_a_path)
        image_b = Image.open(img_b_path)
        image_out = Image.open(img_out_path).convert('1')
        if self.transform:
            image_a, image_b, image_out = self.transform((image_a, image_b, image_out))

        if not self.append_filename:
            return image_a, image_b, image_out
        return image_a, image_b, image_out, self.img_OUT.iloc[idx, 0]
