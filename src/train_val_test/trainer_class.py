import torch
import os
from utils.utils_common import *
import time
from random import sample as SMP
from torch import nn, Tensor
import torch.nn.functional as F
import wandb
import collections
from torchvision.utils import make_grid
import random
from tqdm import tqdm
from utils.recorder import *

def check_zero_params(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Check if any of the filter-wise weights are non-zero
            if torch.nonzero(param).size(0) > 0:
                # print(f'{name} has non-zero weights')
                continue
            else:
                print(f'{name} has all zero weights')
                # print(param.shape)

class Trainer:
    def __init__(self, config, model, device):    
        self.device = device
        self.config = config
        self.model = model
        
        self.optimizer = get_task_optimizer(self.config, self.model)
    
        self.criterion = get_criterion(self.config)
        self.criterion.to(self.device)    
        print('criterion:', self.criterion)
        
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(self.config)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.config, self.train_dataset,self.val_dataset, self.test_dataset)
        from utils.utils_common import ModelWrapper
        self.model_wrapper = ModelWrapper(self.config, self.criterion)        
        # self.task_combinations = get_combinations(self.config)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, min_lr=0.0000001,verbose= True)
        self.inference_time = InferenceTime()
        #if 'lambda_list' in config:
        #    self.lambda_list = config['lambda_list']
        
    def train(self, epoch):   
        
        if (self.config['group_sparsity'] == True) and ('lambda_list' in self.config) and (epoch%10 == 0) and (epoch < len(self.config['lambda_list'])*10):
            idx = int(epoch/10)
            self.config['bb_optimizer_params']['lambda'] = self.config['lambda_list'][idx]
            print(epoch, self.config['bb_optimizer_params']['lambda'])
        
        
        if epoch >= self.config['sparsity_threshold']:
            self.optimizer = get_BB_optimizer(self.config, self.model)
        # else:
        #     optimizer = optimizer1   
        
        
          
        self.model.train()
        losses =  collections.defaultdict(list)
        metric =  collections.defaultdict(lambda: collections.defaultdict(list))
        
        
        
        for i, batch in enumerate(tqdm(self.train_loader)):   
            
            images = batch['image'].to(self.device)
            targets = {task: val.to(self.device) for task, val in batch['targets'].items()} 
            
            if self.config['setup'] == 'singletask':
                output, targets, loss_dict = self.model_wrapper.single_task_forward(self.model,images, targets, 'train')    
            elif self.config['setup'] == 'multitask':
                output, targets, loss_dict = self.model_wrapper.multi_task_forward(self.model, images, targets,'train')           
                       
                        
            self.optimizer.zero_grad()   
            # self.optimizer_decoders.zero_grad()
                     
            loss_dict['total'].backward()           
                     
            self.optimizer.step()
            # self.optimizer_decoders.step()
             
            
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print(name, 'grad is none')
            
            # check_zero_params(self.model)
            
            metric_dict = evaluation_metrics(self.config, output, targets)
            # wandb.watch(self.model, log = 'parameters', log_freq = 5) 
            for task,value in metric_dict.items():
                for k,v in value.items():
                    metric[task][k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for keys, value in loss_dict.items():
                losses[keys].append(value.detach().cpu().numpy())
                

            # if i == 0:
            #     break
        
        losses_ = {task: np.mean(val) for task, val in losses.items()}     
        metric_ = {task: {m: np.mean(val) for m, val in values.items()} for task, values in metric.items()}
        
        sparsity = calculate_percentage_sparsity(self.model)
        wandb.log({"train/sparsity": sparsity})
        
        return losses_,metric_
    
    def validate(self, epoch):
        self.model.eval()
        losses =  collections.defaultdict(list)
        metric =  collections.defaultdict(lambda: collections.defaultdict(list))
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader)):
                images = batch['image'].to(self.device)
                targets = {task: val.to(self.device) for task, val in batch['targets'].items()} 

                if self.config['setup'] == 'singletask':
                    
                    output, targets, loss_dict = self.model_wrapper.single_task_forward(self.model,images, targets,'val')    
                elif self.config['setup'] == 'multitask':
                    
                    output, targets, loss_dict = self.model_wrapper.multi_task_forward(self.model,images, targets, 'val') 

                metric_dict = evaluation_metrics(self.config, output, targets) 

                for task,value in metric_dict.items():
                    for k,v in value.items():
                        metric[task][k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for keys, value in loss_dict.items():
                    losses[keys].append(value.detach().cpu().numpy())
            
                # if i == 0:
                #     break
                
                
            losses_ = {task: np.mean(val) for task, val in losses.items()}     
            metric_ = {task: {m: np.mean(val) for m, val in values.items()} for task, values in metric.items()}
            
            self.scheduler.step(losses_['total'])
            
            # if (epoch %10 == 0) and (self.config['wandb_img_log'] == True):
            #     #### for visulaization on wandb
            #     img_idx = 5
            #     img = images[img_idx,:,:,:]
            #     size = img.shape[-2:]
            #     wandb.log({'input_image': wandb.Image(img)})
            
            
            #     if 'segmentsemantic' in self.config['task_list']:
            #         target = targets['segmentsemantic'][img_idx,0,:,:]
            #         # target = torch.squeeze(target,0)
            #         out = F.interpolate(output['segmentsemantic'], size=size, mode="bilinear")
            #         out = F.softmax(out, dim = 1)
            #         out = out[img_idx,:,:,:]
            #         out = torch.argmax(out, dim = 0)
                    
            #         if self.config['dataset_name'] == 'Taskonomy':
            #             out = draw_segmentation_map_taskonomy(out).permute(2,0,1).float()
            #             tar = draw_segmentation_map_taskonomy(target).permute(2,0,1).float()
            #         elif self.config['dataset_name'] == 'NYU':
            #             out = draw_segmentation_map_NYU(out).permute(2,0,1).float()
            #             tar = draw_segmentation_map_NYU(target).permute(2,0,1).float()
            #         else:
            #             print('dataset not found')
                        
            #         # image_grid = [torch.tensor(tar), torch.tensor(out)]        
            #         image_grid = [tar, out]         
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Segmentation- GT (left) and pred (right)": wandb.Image(grid)})              
            #         # wandb.log({'segmentation_GT': wandb.Image(tar)})
            #         # wandb.log({'segmentation_pred': wandb.Image(out)})  
                    
                    
                    
            #     if 'depth_euclidean' in self.config['task_list']:
            #         tar = F.interpolate(targets['depth_euclidean'], size=size,mode="bilinear")
            #         tar = tar[img_idx,:,:,:]
            #         # wandb.log({'depth_GT': wandb.Image(tar)})
            #         out = F.interpolate(output['depth_euclidean'], size=size, mode="bilinear")
            #         out = out[img_idx,:,:,:]
            #         # out = out.unsqueeze(0)
                    
            #         # image_grid = [torch.tensor(tar), torch.tensor(out)]   
            #         image_grid = [tar, out]                  
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Depth- GT (left) and pred (right)": wandb.Image(grid)})                 
            #         # wandb.log({'depth_pred': wandb.Image(out)})
                    
            #     if  'edge_texture' in self.config['task_list']:
            #         tar1 = F.interpolate(targets['edge_texture'], size=size, mode="bilinear")
            #         tar1 = tar1[img_idx,:,:,:]
            #         # tar = (tar1>0.5).float()
            #         out1 = F.interpolate(output['edge_texture'], size=size, mode="bilinear")
            #         out1 = out1[img_idx,:,:,:]   
            #         # out = (out1>0.5).float()                  
            #         image_grid = [tar1, out1]               
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Edge - GT (left) and pred (right)": wandb.Image(grid)}) 

            #     if 'surface_normal' in self.config['task_list']:
            #         tar = F.interpolate(targets['surface_normal'], size=size, mode="bilinear")
            #         tar = tar[img_idx,:,:,:]
            #         # wandb.log({'SN_GT': wandb.Image(tar)})                   
            #         out = F.interpolate(output['surface_normal'], size=size, mode="bilinear")
            #         out = out[img_idx,:,:,:]
            #         # wandb.log({'SN_pred': wandb.Image(out)})     
            #         # image_grid = [torch.tensor(tar), torch.tensor(out)]   
            #         image_grid = [tar, out]                  
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Surface_normal - GT (left) and pred (right)": wandb.Image(grid)})
                
                
        return losses_,metric_ , self.model


    def test(self, epoch, model):
        epoch= 0
        self.model.eval()
        losses =  collections.defaultdict(list)
        metric =  collections.defaultdict(lambda: collections.defaultdict(list))
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                images = batch['image'].to(self.device)
                targets = {task: val.to(self.device) for task, val in batch['targets'].items()} 

                if self.config['setup'] == 'singletask':
                    
                    output, targets, loss_dict = self.model_wrapper.single_task_forward(model,images, targets, 'test')    
                elif self.config['setup'] == 'multitask':
                    
                    output, targets, loss_dict = self.model_wrapper.multi_task_forward(model,images, targets, 'test') 
                    # if i > 0:
                        # self.inference_time.update(batch_time)
                    
                metric_dict = evaluation_metrics(self.config, output, targets) 

                for task,value in metric_dict.items():
                    for k,v in value.items():
                        metric[task][k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for keys, value in loss_dict.items():
                    losses[keys].append(value.detach().cpu().numpy())
            
                # if i == 0:
                #     break
            # print(self.inference_time.compute())    
            losses_ = {task: np.mean(val) for task, val in losses.items()}     
            metric_ = {task: {m: np.mean(val) for m, val in values.items()} for task, values in metric.items()}
            
            # self.scheduler.step()
            
            if (epoch %10 == 0) and (self.config['wandb_img_log'] == True):
                #### for visulaization on wandb
                img_idx = 5
                img = images[img_idx,:,:,:]
                size = img.shape[-2:]
                wandb.log({'input_image': wandb.Image(img)})
            
            
                if 'segmentsemantic' in self.config['task_list']:
                    target = targets['segmentsemantic'][img_idx,0,:,:]
                    # target = torch.squeeze(target,0)
                    out = F.interpolate(output['segmentsemantic'], size=size, mode="bilinear")
                    out = F.softmax(out, dim = 1)
                    out = out[img_idx,:,:,:]
                    out = torch.argmax(out, dim = 0)
                    
                    if self.config['dataset_name'] == 'Taskonomy':
                        out = draw_segmentation_map_taskonomy(out).permute(2,0,1).float()
                        tar = draw_segmentation_map_taskonomy(target).permute(2,0,1).float()
                    elif self.config['dataset_name'] == 'NYU':
                        out = draw_segmentation_map_NYU(out).permute(2,0,1).float()
                        tar = draw_segmentation_map_NYU(target).permute(2,0,1).float()
                    else:
                        print('dataset not found')
                        
                    # image_grid = [torch.tensor(tar), torch.tensor(out)]        
                    image_grid = [tar, out]         
                    grid = make_grid(image_grid, nrows=2) 
                    wandb.log({"Segmentation- GT (left) and pred (right)": wandb.Image(grid)})              
                    # wandb.log({'segmentation_GT': wandb.Image(tar)})
                    # wandb.log({'segmentation_pred': wandb.Image(out)})  
                    
                    
                    
                if 'depth_euclidean' in self.config['task_list']:
                    tar = F.interpolate(targets['depth_euclidean'], size=size,mode="bilinear")
                    tar = tar[img_idx,:,:,:]
                    # wandb.log({'depth_GT': wandb.Image(tar)})
                    out = F.interpolate(output['depth_euclidean'], size=size, mode="bilinear")
                    out = out[img_idx,:,:,:]
                    # out = out.unsqueeze(0)
                    
                    # image_grid = [torch.tensor(tar), torch.tensor(out)]   
                    image_grid = [tar, out]                  
                    grid = make_grid(image_grid, nrows=2) 
                    wandb.log({"Depth- GT (left) and pred (right)": wandb.Image(grid)})                 
                    # wandb.log({'depth_pred': wandb.Image(out)})
                    
                if  'edge_texture' in self.config['task_list']:
                    tar1 = F.interpolate(targets['edge_texture'], size=size, mode="bilinear")
                    tar1 = tar1[img_idx,:,:,:]
                    # tar = (tar1>0.5).float()
                    out1 = F.interpolate(output['edge_texture'], size=size, mode="bilinear")
                    out1 = out1[img_idx,:,:,:]   
                    # out = (out1>0.5).float()                  
                    image_grid = [tar1, out1]               
                    grid = make_grid(image_grid, nrows=2) 
                    wandb.log({"Edge - GT (left) and pred (right)": wandb.Image(grid)}) 

                if 'surface_normal' in self.config['task_list']:
                    tar = F.interpolate(targets['surface_normal'], size=size, mode="bilinear")
                    tar = tar[img_idx,:,:,:]
                    # wandb.log({'SN_GT': wandb.Image(tar)})                   
                    out = F.interpolate(output['surface_normal'], size=size, mode="bilinear")
                    out = out[img_idx,:,:,:]
                    # wandb.log({'SN_pred': wandb.Image(out)})     
                    # image_grid = [torch.tensor(tar), torch.tensor(out)]   
                    image_grid = [tar, out]                  
                    grid = make_grid(image_grid, nrows=2) 
                    wandb.log({"Surface_normal - GT (left) and pred (right)": wandb.Image(grid)})
                
                
        return losses_,metric_ 


    
    
            
            
            

        






