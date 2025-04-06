import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import matplotlib.colors as mcolors

from datasets import GoProWithEventsDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)


def run(output_dir='visualizations', num_samples=10):
    transform_img = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_events = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = GoProWithEventsDataset(root_dir='data/GOPRO_converted', transform_img=transform_img,
                                     transform_events=transform_events, image_set='train',
                                     num_time_windows=10, crop_size=(500, 500)).get_reduced(num_samples)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, (event_tensor, blur_image, sharp_image) in enumerate(train_loader):
        # Remove batch dimension (cause we use batch size 1 to visualize)
        blur_image = blur_image.squeeze()
        sharp_image = sharp_image.squeeze()
        event_tensor = event_tensor.squeeze(0)
        # Event data only has 2 channels, we add another one to be able to visualize
        blue_channel = np.zeros_like(event_tensor[0, :, :])
        event_image_with_blue = np.stack([event_tensor[0, :, :], event_tensor[1, :, :], blue_channel], axis=-1)

        # Convert black background to white
        mask = np.all(event_image_with_blue == 0, axis=-1)
        event_image_with_blue[mask] = [1, 1, 1]

        # Convert to HWC format
        blur_image_np = blur_image.numpy().transpose(1, 2, 0)
        sharp_image_np = sharp_image.numpy().transpose(1, 2, 0)

        # Only display first and last events frame from the stack
        first_event_image = event_image_with_blue[0]
        last_event_image = event_image_with_blue[-1]

        # Create path to save to
        blur_img_path = dataset.data[idx][0]
        original_image_name = os.path.basename(blur_img_path)
        image_name_without_extension = os.path.splitext(original_image_name)[0]
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{image_name_without_extension}_visualize_{idx:04d}.png")

        fig, axes = plt.subplots(1, 4, figsize=(15, 5))

        axes[0].imshow(blur_image_np)
        axes[0].set_title("Blurry Image")
        axes[0].axis('off')

        axes[1].imshow(sharp_image_np)
        axes[1].set_title("Sharp Image (Ground Truth)")
        axes[1].axis('off')

        axes[2].imshow(first_event_image, norm=mcolors.PowerNorm(gamma=25))
        axes[2].set_title("First Stacked Event Image")
        axes[2].axis('off')

        axes[3].imshow(last_event_image, norm=mcolors.PowerNorm(gamma=25))
        axes[3].set_title("Last Stacked Event Image")
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(path)
