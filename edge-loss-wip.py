import math

import torch
import torch.nn.functional as F


def main():
    data = torch.FloatTensor([
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    ])
    # data = torch.rand(64, 256, 256)

    center_kernel = torch.FloatTensor([[[
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]]])
    region_kernel = torch.FloatTensor([[[
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]]])
    alpha = 0.01
    probs = torch.FloatTensor([
        [
            [0,   0,   0,   0,   0,   0,   0, 0],
            [1,   1,   1, 0.5,   0,   0, 0.5, 0],
            [1,   0,   1, 0.5,   0,   0, 0.5, 0],
            [1,   1,   1, 0.5,   0,   0, 0.5, 0],
            [0,   0,   0,   0,   1,   1,   1, 1],
            [0,   0, 0.9,   0,   0,   0,   0, 0],
            [0,   0,   0,   0,   0,   0,   0, 0],
            [0, 0.5, 0.5, 0.5, 0.5,   0,   0, 0],
        ],
    ])

    probs = F.conv2d(probs, center_kernel)
    center = F.conv2d(data, center_kernel)
    region = F.conv2d(data, region_kernel)
    region_avg = region/8
    center_region_avg_norm = torch.abs(region_avg - center)
    wedge = center_region_avg_norm * alpha
    loss = -wedge * probs
    print("done")
    # data = F.conv2d(data, center_kernel)
    # center = F.conv2d(probs, center_kernel)
    # region = F.conv2d(probs, region_kernel)
    # region_avg = region / 8
    # center_region_avg_norm = torch.abs(center - region_avg)
    # wedge = center_region_avg_norm * alpha
    # loss = -wedge * torch.log2(data)
    # print("done")


if __name__ == '__main__':
    main()
