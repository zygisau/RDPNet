import torch.nn as nn
import torch
import torch.nn.functional as F


class SobelLoss(nn.Module):
    def __init__(self, alpha, device):
        super(SobelLoss, self).__init__()
        self.alpha = alpha
        self.v_kernel = torch.FloatTensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).to(device)
        self.h_kernel = torch.FloatTensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).to(device)

    def forward(self, outputs, labels):
        data_x = F.conv2d(labels, self.v_kernel, padding=1)
        data_y = F.conv2d(labels, self.h_kernel, padding=1)
        gradient_magnitude = torch.sqrt(torch.square(data_x) + torch.square(data_y))
        gradient_magnitude /= gradient_magnitude.max()

        wedge = gradient_magnitude * self.alpha
        loss = -wedge * F.log_softmax(outputs, dim=0)
        return loss.mean()
