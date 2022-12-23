import math

import torch
import torch.nn.functional as F


def main():
    data = torch.FloatTensor([
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ])
    # data = torch.rand(64, 256, 256)

    v_kernel = torch.FloatTensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
    h_kernel = torch.FloatTensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]])
    alpha = 0.05
    probs = torch.FloatTensor([
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
    ])

    data_x = F.conv2d(data, v_kernel, padding=1)
    data_y = F.conv2d(data, h_kernel, padding=1)
    gradient_magnitude = torch.sqrt(torch.square(data_x) + torch.square(data_y))
    gradient_magnitude /= gradient_magnitude.max()
    print(data_x)
    print(data_y)
    print(gradient_magnitude)
    print(gradient_magnitude.shape)
    # region_avg = region/8
    # center_region_avg_norm = torch.abs(region_avg - center)
    wedge = gradient_magnitude * alpha
    loss = -wedge * probs
    print("done")


if __name__ == '__main__':
    main()
