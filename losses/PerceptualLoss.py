import torch
import torchvision.models as models
from torchvision.models import VGG19_Weights


# Taken from https://gist.github.com/brucemuller/37906a86526f53ec7f50af4e77d025c9
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False, normalize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:4].eval())
        blocks.append(models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[4:9].eval())
        blocks.append(models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[9:16].eval())
        blocks.append(models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.resize = resize
        self.normalize = normalize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        if self.normalize:
            input = (input - self.mean) / self.std
            target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss
