import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import datetime

import time

import torch
from tqdm import tqdm

from datasets import GoProWithEventsDataset
from loggers import EvaluationLogger
from models import OurModel
from PIL import Image
import numpy as np

from utils import calculate_ssim, calculate_psnr

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)


def denormalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # If the tensor is batched (B, C, H, W)
    if img_tensor.ndim == 4:
        for channel, m, s in zip(range(img_tensor.shape[1]), mean, std):
            img_tensor[:, channel, :, :] = img_tensor[:, channel, :, :] * s + m
    # For a single image (C, H, W)
    elif img_tensor.ndim == 3:
        for channel, m, s in zip(range(img_tensor.shape[0]), mean, std):
            img_tensor[channel, :, :] = img_tensor[channel, :, :] * s + m

    # Clamp to get rid of artifacts
    img_tensor = torch.clamp(img_tensor, 0, 1)
    return img_tensor


def save_model_outputs_as_images(outputs, original_blurry_image_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Extract the filename from the original blurry image path
    original_image_name = os.path.basename(original_blurry_image_path)
    image_name_without_extension = os.path.splitext(original_image_name)[0]

    for idx, output in enumerate(outputs):
        output = output.permute(1, 2, 0) # (c, h, w) -> (h, w, c)
        output = output.cpu().numpy()
        output = (output * 255).astype(np.uint8)
        output_image = Image.fromarray(output)

        output_image.save(os.path.join(output_dir, f"{image_name_without_extension}_output_{idx:04d}.png"))


def evaluate_model(model, data_loader_test, eval_logger: EvaluationLogger, dataset, output_dir='outputs'):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print(f'Running on {torch.cuda.get_device_name(0)}')
    model.to(device)

    model.load_weights()
    total_start_time = time.time()  # Start timing here

    model.eval()
    outputs = []
    psnr_all = []
    ssim_all = []
    inference_times = []
    with torch.no_grad():
        progress_bar_eval = tqdm(data_loader_test, desc=f"Evaluation")
        for idx, (events_test, images_test, targets_test) in enumerate(progress_bar_eval):
            images = torch.stack([image.to(device) for image in images_test])
            events = torch.stack([event.to(device) for event in events_test])
            targets = torch.stack([target.to(device) for target in targets_test])

            start_inference_time = time.time()
            output = model(images, events, (images.size(2), images.size(3)))
            total_inference_time = time.time() - start_inference_time

            output = denormalize(output)
            images = denormalize(images)
            targets = denormalize(targets)
            # Compute metrics
            psnr_all.append(calculate_psnr(output, targets, crop_border=0, input_order='CHW'))
            ssim_all.append(calculate_ssim(output, targets, crop_border=0, input_order='CHW'))
            inference_times.append(total_inference_time)

            # For eval batch size is 1, so we can do this
            blur_img_path = dataset.data[idx][0]
            save_model_outputs_as_images(output, blur_img_path, output_dir)
            # save_model_outputs_as_images(images, blur_img_path, 'inputs')
            outputs.extend(outputs)

    total_time = time.time() - total_start_time
    tqdm.write(f"Testing completed in {total_time:.2f}s")

    # Save metrics (average them across test instances)
    eval_logger.log_psnr(np.mean(psnr_all))
    eval_logger.log_ssim(np.mean(ssim_all))
    eval_logger.log_inference_time(np.mean(inference_times))

    return outputs


def run(model_path='pretrained_models/our_model.pth', output_dir='outputs'):
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_events = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Remove the get reduced to test on full dataset
    dataset = GoProWithEventsDataset(root_dir='data/GOPRO_converted', transform_img=transform_img, transform_events=transform_events, image_set='test', num_time_windows=10, crop_size=None)
    generator = torch.Generator().manual_seed(42)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)),
                              generator=generator)

    eval_logger = EvaluationLogger(f'logs/test_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    model = OurModel(model_path=model_path)

    evaluate_model(model=model, data_loader_test=test_loader, eval_logger=eval_logger, dataset=dataset, output_dir=output_dir)
