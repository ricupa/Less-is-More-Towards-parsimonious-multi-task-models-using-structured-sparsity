import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
from torch.autograd import Function
import segmentation_models_pytorch as smp
import torchmetrics



class Seg_cross_entropy_loss(Module):
    """
    This function returns cross entropy error for semantic segmentation
    """

    def __init__(self, config):
        super(Seg_cross_entropy_loss, self).__init__()
        
        self.config = config
        if config['seg_loss_fn'] == 'softCEloss':
            self.criterion = smp.losses.SoftCrossEntropyLoss(reduction='mean', smooth_factor=0.1, ignore_index=255, dim=1)

        elif config['seg_loss_fn'] == 'dice':
            self.criterion = smp.losses.DiceLoss(mode = 'multiclass', log_loss=False, from_logits=True, smooth=0.0, ignore_index=255, eps=1e-07)
        elif config['seg_loss_fn'] == 'Tversky':
            self.criterion = smp.losses.TverskyLoss(mode = 'multiclass', from_logits=True, smooth=0.0, ignore_index=255, eps=1e-07, alpha=0.3, beta=0.7, gamma=2.5)
        elif config['seg_loss_fn'] == 'Focal':
            self.criterion = smp.losses.FocalLoss(mode = 'multiclass', alpha=None, gamma=2.5, ignore_index=255, reduction='mean', normalized=False, reduced_threshold=None)
        else:
            print('segmentation loss not found')

    def forward(self, pred, label):

        new_shape = pred.shape[-2:]       
        gt = F.interpolate(label.float(), size=new_shape, mode='nearest')
        gt = torch.squeeze(gt,1)
        assert len(gt.shape) == 3
        assert len(pred.shape) == 4
        gt = gt.type(torch.int64)
        loss = self.criterion(pred, gt)

        return loss



class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure()
        

    def forward(self, pred, target):
        # Convert input tensors to float
        pred = pred.float()
        target = target.float()
        if pred.shape != target.shape:
            new_shape = pred.shape[-2:]
            target = F.interpolate(target, size=new_shape)
        # Compute SSIM
        ssim = self.ssim(pred, target)
        # Compute negative SSIM (to use as a loss function)
        loss = 1 - ssim
        
        return loss

class DepthMSELoss(nn.Module):
    def __init__(self):
        super(DepthMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, gt):
        """
        output: model output
        gt: ground truth
        """
        
        new_shape = output.shape[-2:]
        output = output[:, 0, :, :].unsqueeze(1).cuda()
        
        depth = F.interpolate(gt.float(), size=new_shape, mode='bilinear')
        depth = depth[:, 0, :, :].unsqueeze(1).cuda()

        loss = self.mse_loss(output, depth)

        return loss
    
    
class Depth_combined_loss(nn.Module):
    ''' this is a combination of depth loss, gradient loss and normal loss
    as described in https://arxiv.org/pdf/1803.08673.pdf '''
    def __init__(self, weight=None, size_average=True):
        super(Depth_combined_loss, self).__init__()
        # self.weights = nn.Parameter(torch.zeros((3)))

    def forward(self, output, gt):

        '''
        output : model output
        depth : ground truth
        '''

        new_shape = output.shape[-2:]
        output = output[:,0,:,:].unsqueeze(1).cuda()        

        depth = F.interpolate(gt.float(), size=new_shape, mode='bilinear')
        depth = depth[:,0,:,:].unsqueeze(1).cuda()

        from utils.utils_common import Sobel 
        cos = nn.CosineSimilarity(dim=1, eps=0)

        ones = torch.ones(depth.size(0), 1, depth.size(2),depth.size(3)).float().cuda()
        ones = torch.autograd.Variable(ones)
        
        get_grad = Sobel().cuda()

        depth_grad = get_grad(depth)
        output_grad = get_grad(output)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()

        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()

        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
        loss = loss_depth + loss_normal + (loss_dx + loss_dy)
        
        # loss = torch.tensor(loss, requires_grad= True)   
        return loss


class softmax_cross_entropy_with_softtarget(nn.Module):
    ''' SoftCrossEntropyLoss(input, target) = KLDivLoss(LogSoftmax(input), target)'''
    def __init__(self):
        super(softmax_cross_entropy_with_softtarget, self).__init__()
        

    def forward(self, input, target):

        log_probs = F.log_softmax(input)
        loss_kl = torch.nn.KLDivLoss(reduction='mean')  #mean, batchmean, none
        loss = loss_kl(log_probs, target)
        # loss = torch.tensor(loss, requires_grad= True)   
        return loss





class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.  
    """
    def __init__(self):
        super(DepthLoss, self).__init__()        
        self.loss = nn.L1Loss( reduction='mean')

    def forward(self, out, label):
        # label = label*255
        new_shape = out.shape[-2:]
        label = F.interpolate(label.float(), size=new_shape, mode='bilinear')
        mask = (label != 0)
        loss = self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))
        return loss






class RMSE_log(nn.Module):
    def __init__(self):
        super(RMSE_log, self).__init__()
    
    def forward(self, pred, label):            
        loss = torch.sqrt( torch.mean( torch.abs(torch.log(label)-torch.log(pred)) ** 2 ))
        # loss = torch.tensor(loss, requires_grad= True) 
        return loss


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top

class surface_normal_loss(nn.Module):
    def __init__(self):
        super(surface_normal_loss, self).__init__()
        self.cosine_similiarity = nn.CosineSimilarity()
        self.normalize = Normalize()
    def forward(self, pred, gt): 
        
        new_shape = pred.shape[-2:]        
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)        
        gt = F.interpolate(gt.float(), size=new_shape)
        gt = gt.permute(0,2,3,1).contiguous().view(-1, 3)
           
        labels = (gt.max(dim=1)[0] != 0)
             
        # labels = (gt.min(dim=1)[0] != 1) & (gt.max(dim=1)[0] != 0)
        # labels = (gt.max(dim=1)[0] < 1) & (gt.max(dim=1)[0] != 0)   
                 
        pred = pred[labels]
        gt = gt[labels]
        pred = self.normalize(pred)
        gt = self.normalize(gt)
        # pred = F.normalize(pred,1)
        # gt = F.normalize(gt,1)        
        loss = 1 - self.cosine_similiarity(pred, gt)
        # loss = torch.tensor(loss.mean(), requires_grad= True) 
        return loss.mean()


# class surface_normal_loss(nn.Module):
#     def __init__(self,size_average=True, normalize=False, norm=2):
#         super(surface_normal_loss, self).__init__()

#         self.size_average = size_average

#         if normalize:
#             self.normalize = Normalize()
#         else:
#             self.normalize = None

#         if norm == 1:
#             # print('Using L1 loss for surface normals')
#             self.loss_func = F.l1_loss
#         elif norm == 2:
#             # print('Using L2 loss for surface normals')
#             self.loss_func = F.mse_loss
#         else:
#             raise NotImplementedError

#     def forward(self, out, label, ignore_label=0):  ### out = pred , label = gt
#         assert not label.requires_grad

#         new_shape = out.shape[-2:] 
#         label = F.interpolate(label.float(), size=new_shape)
#         mask = (label != ignore_label)
#         n_valid = torch.sum(mask).item()

#         if self.normalize is not None:
#             out_norm = self.normalize(out)
#             loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')
#         else:
#             loss = self.loss_func(torch.masked_select(out, mask), torch.masked_select(label, mask), reduction='sum')

#         if self.size_average:
#             if ignore_label:
#                 ret_loss = torch.div(loss, max(n_valid, 1e-6))
#                 return ret_loss
#             else:
#                 ret_loss = torch.div(loss, float(np.prod(label.size())))
#                 return ret_loss

     






class edge_loss(nn.Module):
    def __init__(self):
        super(edge_loss, self).__init__()
        # self.l1_loss = nn.L1Loss()
        self.huberloss = nn.HuberLoss(reduction='mean', delta=0.2)  ####For small errors, the loss function uses the L2 norm, and for larger errors, it uses the L1 norm

    def forward(self, pred, gt): 
        new_shape = pred.shape[-2:]
        # pred = torch.sigmoid(pred)
        gt = F.interpolate(gt.float(), size=new_shape)
        # loss = self.l1_loss(pred,gt)
        loss = self.huberloss(pred,gt)
        # loss = torch.tensor(loss.mean(), requires_grad= True) 
        return loss.mean()


# class edge_loss(nn.Module):
#     def __init__(self):
#         super(edge_loss, self).__init__()

#         self.l1loss = nn.L1Loss(reduction='mean')

#     def forward(self, pred, gt): 
#         if pred.shape != gt.shape:
#             new_shape = pred.shape[-2:]        
#             gt = F.interpolate(gt.float(), size=new_shape)
        
#         binary_pred = (pred>0.5).float()  #### as edge detection should be binary
#         binary_gt = (gt>0.5).float()

#         loss = self.l1loss(binary_pred,binary_gt)
#         loss = torch.tensor(loss.mean(), requires_grad= True) 
#         return loss


# class edge_loss(nn.Module):
#     def __init__(self):
#         super(edge_loss, self).__init__()

#         self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')

#     def forward(self, pred, gt): 
#         if pred.shape != gt.shape:
#             new_shape = pred.shape[-2:]        
#             gt = F.interpolate(gt.float(), size=new_shape)
        
#         binary_gt = (gt>0.5).float()
#         loss = self.bceloss(pred,binary_gt)
        
#         return loss