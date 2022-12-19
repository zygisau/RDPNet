import torch.nn as nn
import torch
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self, alpha, device):
        super(EdgeLoss, self).__init__()
        self.alpha = alpha
        self.center_kernel = torch.FloatTensor([[[
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]]]).to(device)
        self.region_kernel = torch.FloatTensor([[[
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]]]).to(device)
        self.center_kernel2d = torch.FloatTensor([[[
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ], [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]]]).to(device)
        self.region_kernel2d = torch.FloatTensor([[[
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ], [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]]]).to(device)
        self.__NUMBER_OF_NEIGHBOURS = 8

    def forward(self, outputs, labels):
        cut_outputs = F.conv2d(outputs, self.center_kernel)

        center = F.conv2d(labels, self.center_kernel)
        region = F.conv2d(labels, self.region_kernel)

        region_avg = region / self.__NUMBER_OF_NEIGHBOURS
        center_region_avg_norm = torch.abs(region_avg - center)

        wedge = center_region_avg_norm * self.alpha
        loss = -wedge * F.log_softmax(cut_outputs, dim=0)
        return loss.mean()
