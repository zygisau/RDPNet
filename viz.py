import os

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from RDPNet import RDPNet
from loader import CDDLoader
from metrics import initialize_metrics
from parser import get_parser_with_args
from transforms import test_transforms

if not os.path.exists('./output_img'):
    os.mkdir('./output_img')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()


device = 'cpu'
if torch.cuda.is_available():
    print("Torch successfully found cuda")
    print("Torch configured to use cuda: ", opt.cuda)
    device = 'cuda'


test_paths = opt.test_paths
test_files = [os.path.join(path, opt.filelist_name) for path in test_paths]

test_data = CDDLoader(test_files, test_paths, transform=test_transforms)
test_dataloader = DataLoader(test_data, batch_size=24, shuffle=True)

def main():
    net = RDPNet(in_ch=3, out_ch=2).to(device)
    net.load_state_dict(torch.load(opt.model_path, map_location=device))
    net.eval()

    index_img = 0
    test_metrics = initialize_metrics()
    with torch.no_grad():
        tbar = tqdm(test_dataloader)
        for batch_img1, batch_img2, labels in tbar:
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.long().to(device)

            cd_preds = net(batch_img1, batch_img2)

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)
            cd_preds = cd_preds.data.cpu().numpy()
            cd_preds = cd_preds.squeeze() * 255

            file_path = './output_img/' + str(index_img).zfill(5)
            cv2.imwrite(file_path + '.png', cd_preds)

            index_img += 1
