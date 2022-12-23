import json
import os

import torch
from numpy import Inf

from RDPNet import RDPNet
from torch.utils.data import DataLoader
import torch.optim as optim
from edge_loss import EdgeLoss
from helpers import get_mean_metrics
from loader import CDDLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support as prfs

from losses import FocalLoss, dice_loss
from metrics import initialize_metrics, set_metrics
from parser import get_parser_with_args
from sobel_loss import SobelLoss
from transforms import train_transforms, test_transforms
from torch.utils.tensorboard import SummaryWriter

models_path = 'tmp_sobel'
isExist = os.path.exists(models_path)
if not isExist:
    os.makedirs(models_path)

parser, metadata = get_parser_with_args()
opt = parser.parse_args()
writer = SummaryWriter('/runs_sobel')

device = 'cpu'
if torch.cuda.is_available():
    print("Torch successfully found cuda")
    print("Torch configured to use cuda: ", opt.cuda)
    device = 'cuda'
    # dev = torch.device('cuda')

train_paths = opt.train_paths
train_files = [os.path.join(path, opt.filelist_name) for path in train_paths]
valid_paths = opt.valid_paths
valid_files = [os.path.join(path, opt.filelist_name) for path in valid_paths]
test_paths = opt.test_paths
test_files = [os.path.join(path, opt.filelist_name) for path in test_paths]

training_data = CDDLoader(train_files, train_paths, transform=train_transforms)
valid_data = CDDLoader(valid_files, valid_paths, transform=test_transforms)
test_data = CDDLoader(test_files, test_paths, transform=test_transforms)
train_dataloader = DataLoader(training_data, batch_size=24, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=24, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=24, shuffle=True)

net = RDPNet(in_ch=3, out_ch=2).to(device)
# net.load_state_dict(torch.load("RDPNet_CDD.pth"))

# criterion1 = EdgeLoss(1, device)
criterion1 = SobelLoss(1, device)
criterion2 = FocalLoss(gamma=0, alpha=None)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.8)

best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}

for epoch in tqdm(range(opt.epochs)):  # loop over the dataset multiple times

    # ----------------------------------
    # -------------TRAINING-------------
    # ----------------------------------
    running_loss = 0.0
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    net.train()
    for i, data in enumerate(tqdm(train_dataloader, leave=False)):
        # get the inputs; data is a list of [inputs, labels]
        inputs = data[:-1]
        inputs = [pic.to(device) for pic in inputs]
        labels = data[-1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(*inputs)
        loss_edge = criterion1(outputs, labels)
        loss_focal = criterion2(outputs, labels)
        loss_dice = dice_loss(outputs, labels)

        loss = loss_edge + loss_focal + loss_dice
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

        writer.add_scalar("Loss/train", loss, epoch)

        _, cd_preds = torch.max(outputs, 1)

        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (outputs.shape[-1] ** 2)))
        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               zero_division=0,
                               pos_label=1)
        train_metrics = set_metrics(train_metrics,
                                    running_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    scheduler.get_last_lr())

        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, epoch)

    scheduler.step()

    # ----------------------------------
    # -----------VALIDATION-------------
    # ----------------------------------
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(valid_dataloader, leave=False)):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[:-1]
            inputs = [pic.to(device) for pic in inputs]
            labels = data[-1].to(device)

            # forward + backward + optimize
            outputs = net(*inputs)
            loss_edge = criterion1(outputs, labels)
            loss_focal = criterion2(outputs, labels)
            loss_dice = dice_loss(outputs, labels)
            loss = loss_edge + loss_focal + loss_dice

            writer.add_scalar("Loss/valid", loss, epoch)

            _, cd_preds = torch.max(outputs, 1)

            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (outputs.shape[-1] ** 2)))
            cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                 cd_preds.data.cpu().numpy().flatten(),
                                 average='binary',
                                 zero_division=0,
                                 pos_label=1)
            val_metrics = set_metrics(val_metrics,
                                      running_loss,
                                      cd_corrects,
                                      cd_val_report,
                                      scheduler.get_last_lr())

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)

            for k, v in mean_val_metrics.items():
                writer.add_scalars(str(k), {'val': v}, epoch)

        if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                or
                (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                or
                (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):
            metadata['validation_metrics'] = mean_val_metrics
            with open('./tmp/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(net.state_dict(), os.path.join('.', models_path, 'checkpoint_cd_epoch_' + str(epoch) + '.pt'))
            best_metrics = mean_val_metrics

    if epoch % (opt.epochs / 4) == 0:
        print("Printing model for safety")
        torch.save(net.state_dict(), os.path.join('.', models_path, 'emergency_checkpoint_cd_epoch_' + str(epoch) + '.pt'))

print('Finished Training')

writer.flush()
writer.close()
