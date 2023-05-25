
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
import segmentation_models_pytorch as smp

       

class calculate_IoU(Module):
    def __init__(self):
        super(calculate_IoU, self).__init__()

    def forward(self,gt, pred):
        with torch.no_grad():
            eps = 1e-8
            n_classes = pred.shape[1]            
            pred = F.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            iou = []
            for gt, pred in zip(gt, pred):
                tp, fp, fn = 0, 0, 0
                valid = (gt != 0)  #### ignore index 0 i.e. background
                for i_part in range(1,n_classes):
                    tmp_gt = (gt == i_part)
                    tmp_pred = (pred == i_part)
                    tp += torch.sum((tmp_gt & tmp_pred & valid).float())
                    fp += torch.sum((~tmp_gt & tmp_pred & valid).float())
                    fn += torch.sum((tmp_gt & ~tmp_pred & valid).float())
                jac = tp / max(tp + fp + fn, eps)
                iou.append(jac)
            # iou = torch.mean(torch.tensor(iou))
            iou = torch.mean(iou)
            
            return iou.cpu().numpy()



            
            
            