import os

import torch
from numpy import Inf

from RDPNet import RDPNet
from torch.utils.data import DataLoader
import torch.optim as optim
from edge_loss import EdgeLoss
from loader import CDDLoader
from tqdm import tqdm

from losses import FocalLoss, dice_loss
from metrics import Evaluator
from parser import get_parser_with_args
from transforms import train_transforms, test_transforms
from torch.utils.tensorboard import SummaryWriter


models_path = 'tmp'
isExist = os.path.exists(models_path)
if not isExist:
    os.makedirs(models_path)

parser, metadata = get_parser_with_args()
opt = parser.parse_args()
writer = SummaryWriter()

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

criterion1 = EdgeLoss(1, device)
criterion2 = FocalLoss(gamma=0, alpha=None)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.8)
evaluator = Evaluator(opt.num_class, opt.cuda)

best_metrics = {
    "mIoU": -Inf,
    "Precision": -Inf,
    "Recall": -Inf,
    "F1": -Inf
}

for epoch in tqdm(range(opt.epochs)):  # loop over the dataset multiple times

    # ----------------------------------
    # -------------TRAINING-------------
    # ----------------------------------
    running_loss = 0.0
    evaluator.reset()
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
        evaluator.add_batch(labels, outputs)

    mIoU = evaluator.Mean_Intersection_over_Union()
    Precision = evaluator.Precision()
    Recall = evaluator.Recall()
    F1 = evaluator.F1()

    writer.add_scalar("mIoU/train", mIoU, epoch)
    writer.add_scalar("Precision/train", Precision, epoch)
    writer.add_scalar("Recall/train", Recall, epoch)
    writer.add_scalar("F1/train", F1, epoch)
    scheduler.step()

    # ----------------------------------
    # -----------VALIDATION-------------
    # ----------------------------------
    evaluator.reset()
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
            evaluator.add_batch(labels, outputs)

        mIoU = evaluator.Mean_Intersection_over_Union()
        Precision = evaluator.Precision()
        Recall = evaluator.Recall()
        F1 = evaluator.F1()
        writer.add_scalar("mIoU/valid", mIoU, epoch)
        writer.add_scalar("Precision/valid", Precision, epoch)
        writer.add_scalar("Recall/valid", Recall, epoch)
        writer.add_scalar("F1/valid", F1, epoch)
        print(f"Epoch validation: {epoch}; mIoU: {mIoU}; Precision: {Precision}; Recall: {Recall}; F1: {F1}")

    """
        Store the weights of good epochs based on validation results
    """
    if ((Precision > best_metrics['Precision'])
            or
            (Recall > best_metrics['Recall'])
            or
            (F1 > best_metrics['F1'])):
        print("Saving new best model...")
        torch.save(net.state_dict(), os.path.join('.', models_path, 'checkpoint_cd_epoch_' + str(epoch) + '.pt'))

print('Finished Training')

writer.flush()
writer.close()
