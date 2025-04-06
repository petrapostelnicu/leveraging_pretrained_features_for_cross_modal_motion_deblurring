import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet34_Weights


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=None):
        super(DecoderBlock, self).__init__()
        # Upsample the input features
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After upsampling, concatenate with skip features then process with convs
        if skip_channels is None:
            self.conv_block = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv_block = nn.Sequential(
                nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x, skip=None):
        x = self.upconv(x)
        if skip is not None:
            # If necessary, crop or pad skip to match x's spatial dims
            if x.shape[2:] != skip.shape[2:]:
                print('here')
                # For simplicity, assuming skip is slightly larger and we center crop it
                diffY = skip.size(2) - x.size(2)
                diffX = skip.size(3) - x.size(3)
                skip = skip[:, :, diffY//2 : diffY//2 + x.size(2), diffX//2 : diffX//2 + x.size(3)]
            x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class OurModel(nn.Module):
    def __init__(self, model_path):
        super(OurModel, self).__init__()

        self.model_path = model_path

        # Encoder 1: Standard 2D ResNet-34 for RGB images
        self.resnet_img = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.image_encoder_conv1 = self.resnet_img.conv1
        self.image_encoder_bn1 = self.resnet_img.bn1
        self.image_encoder_relu = self.resnet_img.relu
        self.image_encoder_maxpool = self.resnet_img.maxpool
        self.image_encoder_layer1 = self.resnet_img.layer1
        self.image_encoder_layer2 = self.resnet_img.layer2
        self.image_encoder_layer3 = self.resnet_img.layer3
        self.image_encoder_layer4 = self.resnet_img.layer4

        # Encoder 2: 2D ResNet-34 for event data with temporal dimension stacked on top of the channels
        self.resnet_events = models.resnet34()
        # The conv1 layer is modified for N-Imagenet and we do it like them
        self.resnet_events.conv1 = nn.Conv2d(
            in_channels=20,
            out_channels=64,
            kernel_size=(14, 14),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )
        self.load_pretrained_weights_events("pretrained_models/best.tar")
        self.events_encoder_conv1 = self.resnet_events.conv1
        self.events_encoder_bn1 = self.resnet_events.bn1
        self.events_encoder_relu = self.resnet_events.relu
        self.events_encoder_maxpool = self.resnet_events.maxpool
        self.events_encoder_layer1 = self.resnet_events.layer1
        self.events_encoder_layer2 = self.resnet_events.layer2
        self.events_encoder_layer3 = self.resnet_events.layer3
        self.events_encoder_layer4 = self.resnet_events.layer4

        # Fusion layer
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder: Upsampling layers (randomly initialized)
        self.decoder_layer1 = DecoderBlock(in_channels=512, out_channels=256, skip_channels=256)
        self.decoder_layer2 = DecoderBlock(in_channels=256, out_channels=128, skip_channels=128)
        self.decoder_layer3 = DecoderBlock(in_channels=128, out_channels=64, skip_channels=128)
        self.decoder_layer4 = DecoderBlock(in_channels=64, out_channels=32)

        # Final conv to go back to 3 channels
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Apply He normal initialization to decoder layers
        self.decoder_layer1.apply(self.init_weights)
        self.decoder_layer2.apply(self.init_weights)
        self.decoder_layer3.apply(self.init_weights)
        self.decoder_layer4.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def build_resnet3d(self):
        """Convert ResNet-34 2D to ResNet-34 3D."""
        resnet2d = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        resnet2d.fc = nn.Identity()
        
        def convert_layer(layer):
            """Recursively convert Conv2d and BatchNorm2d to 3D versions."""
            if isinstance(layer, nn.Conv2d):
                return nn.Conv3d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels,
                    kernel_size=(3, layer.kernel_size[0], layer.kernel_size[1]),  # Convert to 3D
                    stride=(1, layer.stride[0], layer.stride[1]),  # Keep time stride=1
                    padding=(1, layer.padding[0], layer.padding[1]),  # Match padding
                    bias=layer.bias is not None
                )
            elif isinstance(layer, nn.BatchNorm2d):
                return nn.BatchNorm3d(layer.num_features)
            elif isinstance(layer, nn.Sequential):
                return nn.Sequential(*[convert_layer(sub_layer) for sub_layer in layer.children()])
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d):
                return layer  # Keep activation and pooling as is
            else:
                return layer

        def convert_basicblock(block):
            """Convert BasicBlock in ResNet."""
            for name, module in block.named_children():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
                    setattr(block, name, convert_layer(module))
                elif isinstance(module, nn.Sequential) or isinstance(module, nn.Module):
                    convert_basicblock(module)  # Recursively update

        # Convert top-level layers
        resnet2d.conv1 = nn.Conv3d(
            in_channels=2,  # Adjust input channels
            out_channels=64,
            kernel_size=(10, 14, 14),  # Depth of 10
            stride=(1, 2, 2),
            padding=(4, 3, 3),
            bias=False
        )
        resnet2d.bn1 = nn.BatchNorm3d(64)
        resnet2d.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # Convert all residual blocks
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            for block in getattr(resnet2d, layer_name):
                convert_basicblock(block)

        # Convert final pooling layer
        resnet2d.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        return resnet2d


    def load_pretrained_weights_events(self, weight_path):
        """Load 2D pretrained weights and stack time and channel."""
        checkpoint = torch.load(weight_path, weights_only=True)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        # Load existing 2D conv1 weights (Shape: [64, 2, 14, 14])
        old_conv1 = state_dict["conv1.weight"]  # (64, 2, 14, 14)

        # Concat to (64, 20, 14, 14)
        new_conv1 = old_conv1.repeat_interleave(10, dim=1) / 10

        # Load other layers except conv1
        resnet_event_state = self.resnet_events.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in resnet_event_state and k != "conv1.weight"}
        resnet_event_state.update(filtered_state)

        # Assign modified conv1 weights
        with torch.no_grad():
            self.resnet_events.conv1.weight.copy_(new_conv1)

        print("Loaded pretrained 2D weights and correctly stacked time and channels.")

    def forward(self, img_input, event_input, target_size):
        """Forward pass through encoders, fusion, and decoder."""
        # Encode RGB frame
        # print(f'Before image encoder: {img_input.shape}') # (batch_size, 3, 224, 224)
        img_features = self.image_encoder_conv1(img_input)  # (batch_size, 64, 112, 112)
        # print(f'After image encoder conv: {img_features.shape}')
        img_features = self.image_encoder_bn1(img_features) # (batch_size, 64, 112, 112)
        # print(f'After image encoder bn: {img_features.shape}')
        img_features_c1 = self.image_encoder_relu(img_features) # (batch_size, 64, 112, 112)
        # print(f'After image encoder relu: {img_features.shape}')
        img_features = self.image_encoder_maxpool(img_features_c1) # (batch_size, 64, 56, 56)
        # print(f'After image encoder maxpool: {img_features.shape}')
        img_features_1 = self.image_encoder_layer1(img_features) # (batch_size, 64, 56, 56)
        # print(f'After image encoder layer 1: {img_features_1.shape}')
        img_features_2 = self.image_encoder_layer2(img_features_1) # (batch_size, 128, 28, 28)
        # print(f'After image encoder layer 2: {img_features_2.shape}')
        img_features_3 = self.image_encoder_layer3(img_features_2) # (batch_size, 256, 14, 14)
        # print(f'After image encoder layer 3: {img_features_3.shape}')
        # img_features_4 = self.image_encoder_layer4(img_features_3) # (batch_size, 512, 7, 7)
        # print(f'After image encoder layer 4: {img_features_4.shape}')

        # Encode event data
        # print(f'Before event encoder: {event_input.shape}') # (batch_size, 2, 10, 224, 224)
        # Stack channel and temporal dimensions
        event_input = torch.cat([event_input[:, 0], event_input[:, 1]], dim=1) # (batch_size, 20, 224, 224)
        # print(f'Before event encoder, after stacking: {event_input.shape}')
        event_features = self.events_encoder_conv1(event_input)  # (batch_size, 64, 109, 109)
        # print(f'After event encoder conv: {event_features.shape}')
        event_features = self.events_encoder_bn1(event_features) # (batch_size, 64, 109, 109)
        # print(f'After event encoder bn: {event_features.shape}')
        event_features_c1 = self.events_encoder_relu(event_features) # (batch_size, 64, 109, 109)
        # print(f'After event encoder relu: {event_features.shape}')
        event_features = self.events_encoder_maxpool(event_features_c1) # (batch_size, 64, 55, 55)
        # print(f'After event encoder maxpool: {event_features.shape}')
        event_features_1 = self.events_encoder_layer1(event_features) # (batch_size, 64, 55, 55)
        # print(f'After event encoder layer1: {event_features_1.shape}')
        event_features_2 = self.events_encoder_layer2(event_features_1) # (batch_size, 128, 28, 28)
        # print(f'After event encoder layer2: {event_features_2.shape}')
        event_features_3 = self.events_encoder_layer3(event_features_2) # (batch_size, 256, 14, 14)
        # print(f'After event encoder layer3: {event_features_3.shape}')
        # event_features_4 = self.events_encoder_layer4(event_features_3) # (batch_size, 512, 7, 7)
        # print(f'After event encoder layer4: {event_features.shape}')

        # Fuse features
        concatenated_features = torch.cat((img_features_3, event_features_3), dim=1) # (batch_size, 512, 14, 14)
        # print(f'After concat event and image:{concatenated_features.shape}')
        fused_features = self.fusion_conv(concatenated_features)  # (batch_size, 512, 14, 14)
        # print(f'After fusion event and image: {fused_features.shape}')

        # Decode features
        skip_features_2 = torch.cat((img_features_2, event_features_2), dim=1) # (batch_size, 256, 28, 28)
        x = self.decoder_layer1(fused_features, skip_features_2) # (batch_size, 256, 28, 28)
        # print(f'After decoder layer 1: {x.shape}')

        # Here when we fuse features from images and events, we need to get events from 55x55 to 56x56 through interpolation
        skip_features_1 = torch.cat((img_features_1, F.interpolate(event_features_1, size=img_features_1[0, 0, :, :].shape)), dim=1) # (batch_size, 128, 56, 56)
        x = self.decoder_layer2(x, skip_features_1) # (batch_size, 128, 56, 56)
        # print(f'After decoder layer 2: {x.shape}')

        skip_features_c1 = torch.cat((img_features_c1, F.interpolate(event_features_c1, size=img_features_c1[0, 0, :, :].shape)), dim=1) # (batch_size, 128, 112, 112)
        x = self.decoder_layer3(x, skip_features_c1) # (batch_size, 64, 112, 112)
        # print(f'After decoder layer 3: {x.shape}')

        x = self.decoder_layer4(x) # (batch_size, 32, 224, 224)
        # print(f'After decoder layer 4: {x.shape}')

        x = self.final_conv(x) # (batch_size, 3, 224, 224)
        # print(f'After final conv: {x.shape}')

        # Resize the output to match the input image size
        # x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False) # (batch_size, 3, 224, 224)
        # print(f'After interpolation: {x.shape}')

        # Add the initial image to the output (element-wise addition)
        output = x + img_input # (batch_size, 3, 224, 224)
        # print(f'Final output: {output.shape}')

        return output

    def save_weights(self):
        """Save model weights."""
        torch.save(self.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_weights(self, weights_path=None):
        """Load model weights."""
        if weights_path is None:
            path = self.model_path
        else:
            path = weights_path
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
