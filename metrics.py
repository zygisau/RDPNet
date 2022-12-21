import torch
import torch.utils.data
import numpy as np


class Evaluator(object):
    def __init__(self, num_class, cuda=False):
        self.num_class = num_class
        self.cuda = cuda
        # self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.confusion_matrix = torch.zeros(self.num_class, self.num_class)

    def Pixel_Accuracy(self):
        # Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(dim=0).data.cpu().numpy()
        Acc = np.nanmean(Acc)
        # Acc = Acc.mean()
        return Acc

    def Mean_Intersection_over_Union(self):
        # MIoU = np.diag(self.confusion_matrix) / (
        #             np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
        #             np.diag(self.confusion_matrix))
        MIoU = torch.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - torch.diag(self.confusion_matrix)
        ).data.cpu().numpy()
        MIoU = np.nanmean(MIoU)
        # MIoU = MIoU.mean()
        return np.nan_to_num(MIoU, nan=0)

    def Precision(self):
        Pre = self.confusion_matrix[1][1] / (self.confusion_matrix[0][1] + self.confusion_matrix[1][1]).data.cpu().numpy()
        return np.nan_to_num(Pre, nan=0)

    def Recall(self):
        Re = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0]).data.cpu().numpy()
        return np.nan_to_num(Re, nan=0)

    def F1(self):
        Pre = self.confusion_matrix[1][1] / (self.confusion_matrix[0][1] + self.confusion_matrix[1][1])
        Re = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0])
        F1 = 2 * Pre * Re / (Pre+Re)
        return np.nan_to_num(F1, nan=0)

    def Frequency_Weighted_Intersection_over_Union(self):
        # freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        freq = self.confusion_matrix.sum(dim=1) / self.confusion_matrix.sum()
        # iu = np.diag(self.confusion_matrix) / (
        #             np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
        #             np.diag(self.confusion_matrix))
        iu = torch.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) -
            torch.diag(self.confusion_matrix))

        # FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # def _generate_matrix(self, gt_image, pre_image):
    #     mask = (gt_image >= 0) & (gt_image < self.num_class)
    #     label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
    #     count = np.bincount(label, minlength=self.num_class**2)
    #     confusion_matrix = count.reshape(self.num_class, self.num_class)
    #     return confusion_matrix

    def generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        # print(gt_image.device)
        # print(gt_image[mask].device)
        # print(gt_image[mask].type(torch.IntTensor).device)
        # print((self.num_class * gt_image[mask].type(torch.IntTensor)).device)
        # print((self.num_class * gt_image[mask].type(torch.IntTensor) + pre_image[mask]).device)
        if self.cuda:
            label = self.num_class * gt_image[mask].type(torch.IntTensor).cuda() + pre_image[mask]
        else:
            label = self.num_class * gt_image[mask].type(torch.IntTensor) + pre_image[mask]

        tn = (label == 0).type(torch.IntTensor).sum()
        fp = (label == 1).type(torch.IntTensor).sum()
        fn = (label == 2).type(torch.IntTensor).sum()
        tp = (label == 3).type(torch.IntTensor).sum()
        confusion_matrix = torch.tensor([[tn, fp], [fn, tp]])
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert gt_image.shape == pre_image.shape
        pre_image = pre_image[:, [1]]
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self.generate_matrix(gt_image, pre_image)

    def reset(self):
        # self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.confusion_matrix = torch.zeros(self.num_class, self.num_class)


def initialize_metrics():
    """Generates a dictionary of metrics with metrics as keys
       and empty lists as values

    Returns
    -------
    dict
        a dictionary of metrics

    """
    metrics = {
        'cd_losses': [],
        'cd_corrects': [],
        'cd_precisions': [],
        'cd_recalls': [],
        'cd_f1scores': [],
        'learning_rate': [],
    }

    return metrics


def set_metrics(metric_dict, cd_loss, cd_corrects, cd_report, lr):
    """Updates metric dict with batch metrics

    Parameters
    ----------
    metric_dict : dict
        dict of metrics
    cd_loss : dict(?)
        loss value
    cd_corrects : dict(?)
        number of correct results (to generate accuracy
    cd_report : list
        precision, recall, f1 values

    Returns
    -------
    dict
        dict of  updated metrics


    """
    metric_dict['cd_losses'].append(cd_loss)
    metric_dict['cd_corrects'].append(cd_corrects.item())
    metric_dict['cd_precisions'].append(cd_report[0])
    metric_dict['cd_recalls'].append(cd_report[1])
    metric_dict['cd_f1scores'].append(cd_report[2])
    metric_dict['learning_rate'].append(lr)

    return metric_dict
