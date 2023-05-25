import numpy as np
import math
import torch



def get_depth_metric(pred, gt, binary_mask):
    
    depth_output_true = pred.masked_select(binary_mask)
    depth_gt_true = gt.masked_select(binary_mask)
    abs_err = torch.abs(depth_output_true - depth_gt_true)
    rel_err = (torch.abs(depth_output_true - depth_gt_true)+ 1e-7) / (depth_gt_true + 1e-5)
    sq_rel_err = torch.pow(depth_output_true - depth_gt_true, 2) / depth_gt_true
    abs_err = torch.sum(abs_err) / torch.nonzero(binary_mask).size(0)
    # abs_err = torch.mean(abs_err)
    rel_err = torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)
    sq_rel_err = torch.sum(sq_rel_err) / torch.nonzero(binary_mask).size(0)
    # calcuate the sigma
    term1 = depth_output_true / depth_gt_true
    term2 = depth_gt_true / depth_output_true
    ratio = torch.max(torch.stack([term1, term2], dim=0), dim=0)
    # calcualte rms
    rms = torch.mean(torch.pow(depth_output_true - depth_gt_true, 2))
    rms_log = torch.pow(torch.log10(depth_output_true + 1e-7) - torch.log10(depth_gt_true + 1e-7), 2)

    return abs_err.detach().cpu().numpy(), rel_err.detach().cpu().numpy(), sq_rel_err.detach().cpu().numpy(), ratio[0].detach().cpu().numpy(), rms.detach().cpu().numpy(), rms_log.detach().cpu().numpy()
