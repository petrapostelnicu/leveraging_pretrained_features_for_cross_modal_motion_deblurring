import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class GoProDataset(Dataset):
    def __init__(self, root_dir, transform=None, image_set='Train'):
        self.root_dir = root_dir
        self.mode = image_set
        self.transform = transform

        self.data_dir = os.path.join(self.root_dir, self.mode)

        self.scenes = [os.path.join(self.data_dir, scene) for scene in os.listdir(self.data_dir) if
                       os.path.isdir(os.path.join(self.data_dir, scene))]

        self.image_pairs = []

        for scene in self.scenes:
            blur_dir = os.path.join(scene, 'blur')
            sharp_dir = os.path.join(scene, 'sharp')

            blur_images = sorted([f for f in os.listdir(blur_dir) if f.endswith('.png')])
            sharp_images = sorted([f for f in os.listdir(sharp_dir) if f.endswith('.png')])

            for blur_img, sharp_img in zip(blur_images, sharp_images):
                self.image_pairs.append((os.path.join(blur_dir, blur_img), os.path.join(sharp_dir, sharp_img)))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        blur_img_path, sharp_img_path = self.image_pairs[idx]

        blur_image = Image.open(blur_img_path).convert('RGB')
        sharp_image = Image.open(sharp_img_path).convert('RGB')

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, sharp_image

    def get_train_val_datasets(self, val_split=0.2):
        train_size = int((1 - val_split) * len(self))
        val_size = len(self) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(self, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset

    def get_reduced(self, size):
        rest = len(self) - size
        generator = torch.Generator().manual_seed(42)
        dataset, _ = random_split(self, [size, rest], generator=generator)
        return dataset