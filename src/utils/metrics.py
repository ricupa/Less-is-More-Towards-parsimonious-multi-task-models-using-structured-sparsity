import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
# from torchmetrics import IoU,JaccardIndex
import torch
import torch.nn.functional as F
import torch.nn as nn

def calculate_f1score(pred,gt):
    f1 = f1_score(gt, pred, average=None, zero_division = 1)
    return f1.mean()


def calculate_precision(pred,gt):
    score = precision_score(gt, pred, average=None, zero_division = 1)
    return score.mean()


def calculate_recall(pred,gt):
    score = recall_score(gt, pred, average=None, zero_division = 1)
    return score.mean()


def sn_metrics(pred, gt):
    pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)

    labels = (gt.max(dim=1)[0] != 0) 
    # labels = (gt.max(dim=1)[0] < 1) & (gt.max(dim=1)[0] != 0)   
    gt = gt[labels]
    pred = pred[labels]

    gt = F.normalize(gt.float(), dim=1)
    pred = F.normalize(pred, dim=1)
    cosine_similiarity = nn.CosineSimilarity()
    cos_similarity = cosine_similiarity(gt, pred).detach().cpu().numpy()

    overall_cos = np.clip(cos_similarity, -1, 1)
    all_angles = np.arccos(overall_cos) / np.pi * 180.0  
    return overall_cos.mean(), all_angles



def edge_metrics(pred, gt):
    # pred = (pred>0.4).float()  #### as edge detection should be binary
    
    # gt = (gt>0.5).float()
    
    binary_mask = (gt != 0)  ### ignore 0 pixels 
    edge_output_true = pred.masked_select(binary_mask)
    edge_gt_true = gt.masked_select(binary_mask)
    abs_err = torch.abs(edge_output_true - edge_gt_true)  
    
    # fn = nn.L1Loss(reduction='mean')
    # abs_err = fn(edge_output_true,edge_gt_true)
    abs_err = abs_err.mean()
    return abs_err.detach().cpu().numpy()

