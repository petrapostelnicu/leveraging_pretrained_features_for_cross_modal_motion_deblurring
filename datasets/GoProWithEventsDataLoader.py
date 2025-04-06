import os
import random

import cv2
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class GoProWithEventsDataset(Dataset):
    def __init__(self, root_dir, transform_img=None, transform_events=None, image_set='train', num_time_windows=10, crop_size=(224, 224)):
        self.root_dir = root_dir
        self.mode = image_set
        self.transform_img = transform_img
        self.transform_events = transform_events
        self.num_time_windows = num_time_windows
        self.crop_size = crop_size

        self.data_dir = os.path.join(self.root_dir, self.mode)
        self.scenes = [os.path.join(self.data_dir, scene) for scene in os.listdir(self.data_dir) if
                       os.path.isdir(os.path.join(self.data_dir, scene))]

        self.data = []
        for scene in self.scenes:
            blur_dir = os.path.join(scene, 'blur')
            sharp_dir = os.path.join(scene, 'sharp')
            events_dir = os.path.join(scene, 'events_img')

            blur_images = sorted([f for f in os.listdir(blur_dir) if f.endswith('.png')])
            sharp_images = sorted([f for f in os.listdir(sharp_dir) if f.endswith('.png')])

            all_events = sorted([f for f in os.listdir(events_dir) if f.endswith('.png')])

            # Group the event images into stacks of 10
            event_stacks = []
            for i in range(0, len(all_events), self.num_time_windows):
                stack = all_events[i:i + self.num_time_windows]
                event_stacks.append([os.path.join(events_dir, f) for f in stack])

            for blur_img, sharp_img, events in zip(blur_images, sharp_images, event_stacks):
                self.data.append((os.path.join(blur_dir, blur_img), os.path.join(sharp_dir, sharp_img), events))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        blur_img_path, sharp_img_path, events_path = self.data[idx]

        blur_image = Image.open(blur_img_path).convert('RGB')
        sharp_image = Image.open(sharp_img_path).convert('RGB')
        # event_tensor = self.load_event_images(events_path)
        # (h, w, c) -> (c, h, w)
        # event_tensor = event_tensor.permute(0, 3, 1, 2)
        # Load the stack of 10 event images
        event_images = []
        for event_img_path in events_path:
            # event_img = cv2.imread(event_img_path)
            # event_img = cv2.cvtColor(event_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            event_img = Image.open(event_img_path).convert('RGB')
            event_images.append(event_img)

        if self.crop_size is not None:
            blur_image, sharp_image, event_images = self.random_crop_with_event_area(blur_image, sharp_image, event_images)

        event_tensor = []
        for event_img in event_images:
            if self.transform_events:
                event_img = self.transform_events(event_img)

            # Identify and remove the empty channel (which should have zeros)
            # non_zero_channels = torch.sum(event_img, dim=(1, 2)) > 0.0
            # non_zero_channels = ~torch.all(event_img == 0.0, dim=(1, 2))
            # Hardcode empty channel (Red), because in some center crop images there is also another channel that has 0's everywhere
            non_zero_channels = [False, True, True]
            event_img = event_img[non_zero_channels, :, :]

            event_tensor.append(event_img)

        event_tensor = torch.stack(event_tensor, dim=0)

        # Apply the transformation to resize both images and event tensor
        if self.transform_img:
            # print(f'Image before:{blur_image.size}')
            blur_image = self.transform_img(blur_image)
            # print(f'Image after:{blur_image.shape}')
            sharp_image = self.transform_img(sharp_image)

            # Resize event tensor to match the image size (assuming the image is resized)
            # c, h, w = blur_image.shape
            # event_tensor = self.resize_event_tensor(event_tensor, [h, w])

        return event_tensor.permute(1, 0, 2, 3), blur_image, sharp_image

    def load_event_images(self, event_img_paths):
        # Load the stack of 10 event images
        event_tensor = []
        for event_img_path in event_img_paths:
            # event_img = cv2.imread(event_img_path)
            # event_img = cv2.cvtColor(event_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            event_img = Image.open(event_img_path).convert('RGB')


            if self.transform_events:
                event_img = self.transform_events(event_img)

            # Identify and remove the empty channel (which should have zeros)
            # non_zero_channels = torch.sum(event_img, dim=(1, 2)) > 0.0
            # non_zero_channels = ~torch.all(event_img == 0.0, dim=(1, 2))
            # Hardcode empty channel (Red), because in some center crop images there is also another channel that has 0's everywhere
            non_zero_channels = [False, True, True]
            event_img = event_img[non_zero_channels, :, :]

            event_tensor.append(event_img)

        event_tensor = torch.stack(event_tensor, dim=0)

        return event_tensor

    def resize_event_tensor(self, event_tensor, target_size):
        resize_transform = transforms.Resize((target_size[0], target_size[1]))

        # Resize image
        event_tensor_resized = resize_transform(event_tensor)

        return event_tensor_resized

    def random_crop_with_event_area(self, blur_image, sharp_image, event_images):
        masks = []
        for ev in event_images:
            ev_np = np.array(ev)
            mask = ((ev_np[:, :, 1] > 0) | (ev_np[:, :, 2] > 0)).astype(np.uint8)
            masks.append(mask)
        union_mask = (np.any(np.stack(masks, axis=0), axis=0).astype(np.uint8)) * 255

        # Find external contours in the union mask
        contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        width, height = blur_image.size
        crop_w, crop_h = self.crop_size

        # if no contours are found, just take a random crop over the full image
        if len(contours) == 0:
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            return (
                TF.crop(blur_image, top, left, crop_h, crop_w),
                TF.crop(sharp_image, top, left, crop_h, crop_w),
                [TF.crop(ev, top, left, crop_h, crop_w) for ev in event_images]
            )

        # Get bounding boxes for each contour
        bboxes = [cv2.boundingRect(cnt) for cnt in contours]
        # Randomly choose one bounding box
        bbox = random.choice(bboxes)
        x, y, w_box, h_box = bbox

        # Determine valid horizontal range for the crop such that the bbox is fully contained
        left_min = x + w_box - crop_w
        left_max = x
        left_min = max(left_min, 0)
        left_max = min(left_max, width - crop_w)

        if left_min > left_max:
            left = max(0, min(x, width - crop_w))
        else:
            left = random.randint(left_min, left_max)

        # Determine valid vertical range
        top_min = y + h_box - crop_h
        top_max = y
        top_min = max(top_min, 0)
        top_max = min(top_max, height - crop_h)

        if top_min > top_max:
            top = max(0, min(y, height - crop_h))
        else:
            top = random.randint(top_min, top_max)

        blur_crop = TF.crop(blur_image, top, left, crop_h, crop_w)
        sharp_crop = TF.crop(sharp_image, top, left, crop_h, crop_w)
        event_crops = [TF.crop(ev, top, left, crop_h, crop_w) for ev in event_images]

        return blur_crop, sharp_crop, event_crops

    def get_train_val_datasets(self, val_split=0.2):
        train_size = int((1 - val_split) * len(self))
        val_size = len(self) - train_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(self, [train_size, val_size], generator=generator)
        return train_dataset, val_dataset

    def get_reduced(self, size):
        # selected_pairs = self.data[:size]
        random.seed(42)
        selected_pairs = random.sample(self.data, size)
        reduced_dataset = GoProWithEventsDataset(self.root_dir, self.transform_img, self.transform_events, self.mode, self.num_time_windows, self.crop_size)
        reduced_dataset.data = selected_pairs
        return reduced_dataset
