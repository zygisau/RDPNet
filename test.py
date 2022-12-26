import os

import torch
from torch.utils.data import DataLoader

from RDPNet import RDPNet
from helpers import get_mean_metrics
from loader import CDDLoader
from metrics import set_metrics, initialize_metrics
from parser import get_parser_with_args
from tqdm import tqdm
from transforms import test_transforms
from sklearn.metrics import precision_recall_fscore_support as prfs

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
    test_metrics = initialize_metrics()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader, leave=False)):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[:-1]
            inputs = [pic.to(device) for pic in inputs]
            labels = data[-1].to(device)

            # forward + backward + optimize
            outputs = net(*inputs)
            _, cd_preds = torch.max(outputs, 1)

            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (outputs.shape[-1] ** 2)))
            cd_test_report = prfs(labels.data.cpu().numpy().flatten(),
                                  cd_preds.data.cpu().numpy().flatten(),
                                  average='binary',
                                  zero_division=0,
                                  pos_label=1)
            test_metrics = set_metrics(test_metrics,
                                       0,
                                       cd_corrects,
                                       cd_test_report, 0)

        mean_val_metrics = get_mean_metrics(test_metrics)
        print(f'cd_corrects: {mean_val_metrics["cd_corrects"]}',
              f'cd_precisions: {mean_val_metrics["cd_precisions"]}',
              f'cd_recalls: {mean_val_metrics["cd_recalls"]}',
              f'cd_f1scores: {mean_val_metrics["cd_f1scores"]}')


if __name__ == "__main__":
    main()
