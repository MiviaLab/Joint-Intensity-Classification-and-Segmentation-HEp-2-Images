import torch.nn as nn
import torchgeometry.losses as losses
import torch.nn.functional as F
import torch
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        
        inputs = torch.sigmoid(inputs)
        
        N = targets.size(0)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = ((2. * intersection ) / (inputs.sum() + targets.sum() + smooth)) / N
        
        return 1 - dice


class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.dice_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')
        self.bce_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')

    def forward(self, preds, mask, intensity):
        binaryCrossEntropy = nn.BCEWithLogitsLoss()
        diceLoss = DiceLoss()

        intensity = intensity.unsqueeze(1)
        intensity = intensity.float()

        loss0 = diceLoss(preds[0], mask)
        loss1 = binaryCrossEntropy(preds[1], intensity)
        
        self.dice_loss += loss0
        self.bce_loss += loss1
        
        loss_0 = (1/2) * loss0
        loss_1 = (1/2) * loss1
        
        
        return loss_0 + loss_1

    def get_losses(self, c):
        return self.dice_loss.item()/c, self.bce_loss.item()/c
    
    def set_losses(self):
        self.dice_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')
        self.bce_loss = torch.as_tensor(0, dtype=torch.float32, device='cuda')