import torch.nn as nn
import torch.nn.functional as F
from .PerceptualLoss import VGGPerceptualLoss
from .CharbonnierLoss import CharbonnierLoss


class OurLoss(nn.Module):
    def __init__(self, perceptual_weight=0.05, normalize=False, use_charbonnier=False):
        super(OurLoss, self).__init__()
        self.perceptual_loss = VGGPerceptualLoss(normalize=normalize)
        self.perceptual_weight = perceptual_weight
        self.charbonnier_loss = CharbonnierLoss()
        self.use_charbonnier = use_charbonnier

    def forward(self, predicted, target):
        perceptual_loss = self.perceptual_loss(predicted, target)
        if self.use_charbonnier is True:
            charbonnier_loss = self.charbonnier_loss(predicted, target)
            total_loss = charbonnier_loss + self.perceptual_weight * perceptual_loss
        else:
            l1_loss = F.l1_loss(predicted, target)
            total_loss = l1_loss + self.perceptual_weight * perceptual_loss
        return total_loss
