

import os
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"]= '1'
# os.enviorn['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:20000'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import numpy as np
import sys
# import torch
from termcolor import colored
import pandas as pd
import collections
import time
import yaml
import argparse
import dill as pickle
import warnings
warnings.filterwarnings('ignore')
# from torchsummary import summary
from utils.pytorchtools import EarlyStopping
from utils.utils_common import *
from train_val_test.trainer_class import Trainer
import wandb
import json
from proxssi.groups.resnet_gp import resnet_groups
from proxssi.optimizers.adamw_hf import AdamW
# from proxssi.tests import penalties
from proxssi import penalties


# from torch.utils.tensorboard import SummaryWriter
# import json
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--config_exp', help='Config file for the experiment')
args = parser.parse_args()




def check_zero_params(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Check if any of the filter-wise weights are non-zero
            if torch.nonzero(param).size(0) > 0:
                # print(f'{name} has non-zero weights')
                continue
            else:
                print(f'{name} has all zero weights')

def main():
    global device 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    config = create_config( args.config_exp) 
    config['device'] = device
    
    task_early_stopping = {}  
    
    lambda_candidates = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    
    config['best_lambda'] = None
    
    for i, lambda_ in enumerate(lambda_candidates):
        print('lambda_candidate :', lambda_)
        
        config['bb_optimizer_params']['lambda'] = torch.tensor(lambda_)
            
        name = config['Experiment_name'] + '_lambda_'+ str(lambda_)
        
        print('Experiment_name: ', name)

        fname = config['checkpoint_folder']+ name

        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            print('folder already exist')           
        
        runs = wandb.init(project= 'MTL_sparsity',
                          name= name, entity='write_entity_name', 
                          config=config, 
                          dir = fname,
                          reinit=True)
        
        wandb.config.update(config, allow_val_change=True)    
        
        with open(fname+'/'+'config_file.yaml','w')as file:
            doc = yaml.dump(config,file)              
        
        early_stopping = EarlyStopping(patience=config['earlystop_patience'], verbose=True, path=fname+'/'+ 'checkpoint.pt')
        
        task_early_stopping = {}
        for task in config['task_list']:
            task_early_stopping[task] = EarlyStopping(patience=config['task_earlystop_patience'], verbose=True, path=fname+'/'+ task+'_checkpoint.pt')
            config['es_loss'] = {}
            
                               
            
        model = get_model(config)   ##### write a get_model function in utils_common
        start_epoch = 0  
        epochwise_train_losses_dict = collections.defaultdict(list)
        epochwise_val_losses_dict = collections.defaultdict(list)
        epochwise_train_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))   # for nested dictionary
        epochwise_val_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))
        training_dict = {}
        validation_dict = {}
        model = model.to(device)
        
        
        # model = copy.deepcopy(model)
        trainer = Trainer(config, model, device) 

        for epoch in range(start_epoch, config['epochs']):
            print(colored('Epoch %d/%d' %(epoch, config['epochs']-1), 'yellow'))
            print(colored('-'*10, 'yellow'))                   
              
            
            loss, metric = trainer.train(epoch)  ##### train the model
            print('train loss: ', loss)
            print('train metric: ', metric)
                    
            for key, value in loss.items():
                wandb.log({f"train/loss/{key}": value})
                epochwise_train_losses_dict[key].append(float(value))

            
            for key, value in metric.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        wandb.log({f"train/metric/{key}/{sub_key}": sub_value})
                        epochwise_train_metric_dict[key][sub_key].append(float(sub_value))
                else:
                    wandb.log({f"train/metric/{key}": value})
                    epochwise_train_metric_dict[key].append(float(value))
            # 
            ##### validate the model 
            vloss, vmetric, model = trainer.validate(epoch)  #### validate the model
            print('val loss: ', vloss)
            print('val metric: ', vmetric)
            
            
            
            for key, value in vloss.items():
                wandb.log({f"validation/loss/{key}": value})
                epochwise_val_losses_dict[key].append(float(value))
                
            for key, value in vmetric.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        wandb.log({f"validation/metric/{key}/{sub_key}": sub_value})
                        epochwise_val_metric_dict[key][sub_key].append(float(sub_value))
                else:
                    wandb.log({f"validation/metric/{key}": value})
                    epochwise_val_metric_dict[key].append(float(value))      
                            
            
            
                    
            optimizer = get_BB_optimizer(config, model.backbone) #### no use of this 
            if config['setup'] == 'multitask':
                for task, loss in vloss.items():
                    if (task != 'total') and (config['flag'][task] == 1):
                        task_early_stopping[task](loss, model, epoch, optimizer, model_checkpoint=True)
                        if task_early_stopping[task].early_stop:
                            print("Early stopping task -", task)
                            config['flag'][task] = 0
                            config['es_loss'][task] = loss  ### no use 
                            head = model.decoders[task]
                            #### freeze the task head and backbone
                            for params in head.parameters():
                                params.requires_grad = False


            

            early_stopping(vloss['total'], model, epoch, optimizer, model_checkpoint=True)
            if (sum(config['flag'].values()) == 0 ) or (early_stopping.early_stop == True):
                print("Early stopping")
                break
            

        model = get_model(config)
        model = model.cuda()
        epoch = 0
        
        # if config['setup'] == 'singletask':
        #     print('singletask')
        checkpoint = torch.load(fname + '/checkpoint.pt')
        model.load_state_dict(checkpoint['model'])
        test_loss, test_metric = trainer.test(epoch, model)
        print('avg_test_metric:', test_metric)
        spar, sparsity_params, total_params, zero_params = calculate_percentage_sparsity_params(model)
        
        print('sparsity :', spar)
        print('sparsity_params:', sparsity_params)
        print('total_params:', total_params)
        print('zero_params:', zero_params)
        
        # else:
        #     test_metric = {}
        #     sp = []
        #     checkpoint = torch.load(fname + '/checkpoint.pt')
        #     model.load_state_dict(checkpoint['model'])
            
        #     for task in config['task_list']:
        #         task_chkpt = fname + '/' + task + '_checkpoint.pt'
        #         if os.path.exists(task_chkpt):
        #             print('task checkpoint exists')
        #             checkpoint = torch.load(task_chkpt)
        #             model.load_state_dict(checkpoint['model'])
        #         else:
        #             print('task checkpoint not found')
                
        #         loss, metric = trainer.test(epoch, model)
        #         test_metric[task] = metric[task]
                
        #         sp.append(calculate_percentage_sparsity(model))
            
        #     print('avg_test_metric:', test_metric) 
        #     sparsity = np.mean(sp)
                
        
        if i == 0:
                config['best_lambda'] = lambda_
                best_metrics = test_metric
        else:
            L = find_best_metric(config, best_metrics, test_metric, config['best_lambda'], lambda_, spar)
            print('best_lambda: ', L)
            config['best_lambda'] = L

        del model
        del trainer


    #### save the configs for future use 
    with open(fname+'/'+'config_file.yaml','w')as file:
        doc = yaml.dump(config,file)
    
    
    print('best_lambda = ', config['best_lambda'])
        
    
    





if __name__ == "__main__":
    main()
