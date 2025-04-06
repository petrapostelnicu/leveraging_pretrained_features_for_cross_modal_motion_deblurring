import os
import sys

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import datetime

import time

import torch
import torch.optim as optim
from tqdm import tqdm

from datasets import GoProWithEventsDataset
from loggers import LossLogger
from losses import OurLoss
from models import OurModel

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)


def train_model(model, data_loader_train, data_loader_val, loss_logger: LossLogger, num_epochs=50, freeze_encoder_weights=False, lr=0.002, patience=10, load_pretrained=False, use_charbonnier_loss=False, perceptual_loss_weight=0.05):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print(f'Running on {torch.cuda.get_device_name(0)}')
    model.to(device)

    if load_pretrained is True:
        model.load_weights()
        print(f'Loaded weights from {model.model_path}')

    total_start_time = time.time()  # Start timing here
    model.train()

    if freeze_encoder_weights is True:
        # Freeze the weights for the image encoder
        for param in model.resnet_img.parameters():
            param.requires_grad = False

        # Freeze the weights for the event encoder
        for param in model.resnet_event.parameters():
            param.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    criterion = OurLoss(use_charbonnier=use_charbonnier_loss, perceptual_weight=perceptual_loss_weight).to(device)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    # Training
    for epoch in range(num_epochs):
        progress_bar_train = tqdm(data_loader_train, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
        train_running_loss = 0.0
        for train_events, train_images, train_targets in progress_bar_train:
            images = torch.stack([image.to(device) for image in train_images])
            events = torch.stack([event.to(device) for event in train_events])
            targets = torch.stack([target.to(device) for target in train_targets])

            output = model(images, events, (images.size(2), images.size(3)))
            loss = criterion(output, targets)
            train_running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_train_loss = train_running_loss / len(data_loader_train)
        train_losses.append(average_train_loss)
        progress_bar_train.close()
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Training - Loss: {average_train_loss}")

        # Validation
        progress_bar_val = tqdm(data_loader_val, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation")
        val_running_loss = 0.0
        with torch.no_grad():
            for val_events, val_images, val_targets in progress_bar_val:
                images = torch.stack([image.to(device) for image in val_images])
                events = torch.stack([event.to(device) for event in val_events])
                targets = torch.stack([target.to(device) for target in val_targets])

                output = model(images, events, (images.size(2), images.size(3)))
                loss = criterion(output, targets)
                val_running_loss += loss.item()

        average_val_loss = val_running_loss / len(data_loader_val)
        val_losses.append(average_val_loss)
        progress_bar_val.close()
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Validation - Loss: {average_val_loss}")

        # Check for improvement
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            model.save_weights()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # If no improvement for a certain number of epochs, restore best weights and reduce learning rate
        if epochs_since_improvement >= patience:
            tqdm.write(
                f"Validation loss did not improve for {patience} epochs. Reducing learning rate and restoring best model weights.")
            model.load_weights()
            model.to(device)
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
            epochs_since_improvement = 0

    total_time = time.time() - total_start_time
    tqdm.write(f"Training completed in {total_time:.2f}s")
    loss_logger.log_all_losses(num_epochs=num_epochs, train_losses=train_losses, val_losses=val_losses)
    loss_logger.log_training_time(total_time)

def run(config):
    # Load configs
    model_path = config['MODEL']['model_path']
    batch_size = int(config['TRAINING']['batch_size'])
    num_epochs = int(config['TRAINING']['num_epochs'])
    freeze_encoder_weights = config['TRAINING']['freeze_encoder_weights']
    lr = float(config['TRAINING']['lr'])
    patience = int(config['TRAINING']['patience'])
    load_pretrained = config['TRAINING']['load_pretrained']
    use_charbonnier_loss = config['LOSS']['use_charbonnier_loss']
    perceptual_loss_weight = float(config['LOSS']['perceptual_loss_weight'])

    transform_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_events = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Remove the get reduced to train on full dataset
    dataset = GoProWithEventsDataset(root_dir='data/GOPRO_converted', transform_img=transform_img, transform_events=transform_events, image_set='train', num_time_windows=10)
    train_dataset, val_dataset = dataset.get_train_val_datasets(val_split=0.2)
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)),
                              generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)),
                            generator=generator)

    loss_logger = LossLogger(f'logs/train_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    model = OurModel(model_path=model_path)

    train_model(model=model, data_loader_train=train_loader, data_loader_val=val_loader, num_epochs=num_epochs,
                      loss_logger=loss_logger, freeze_encoder_weights=freeze_encoder_weights, lr=lr, patience=patience, load_pretrained=load_pretrained, use_charbonnier_loss=use_charbonnier_loss, perceptual_loss_weight=perceptual_loss_weight)
