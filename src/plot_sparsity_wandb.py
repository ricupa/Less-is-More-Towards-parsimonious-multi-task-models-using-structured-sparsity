
##### conda activate MTLenv
#### python plot_sparsity_wandb.py --exp_folder_path /home/ricupa/Documents/MTL_meta_adaptive_features/MTL_adaptive_results/new/3_single_class_male_1e5_trial_1/

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
# import collections
# import time
import yaml
import argparse
import dill as pickle
import warnings
warnings.filterwarnings('ignore')
# from torchsummary import summary
from utils.utils_common import *
from dataset.create_dataset import NYUDataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import matplotlib.patches as mpatches
# from flopth import flopth

parser = argparse.ArgumentParser(description='Plot_sparsity')
parser.add_argument('--exp_folder_path', help='Config file for the experiment')
args = parser.parse_args()



def get_layer_name(lname):
    layer_name = []
    for i, name in enumerate(lname):
        temp = name.split('.')
        temp1 = temp[1:-1]
        n = '.'.join(temp1)
        layer_name.append(n)
    return layer_name

def check_zero_params(model, config):
    flag = []
    layername = []
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('weight' in name) and ('conv' in name) and (len(param.shape) == 4):
            layername.append(name)
            if torch.nonzero(param).size(1) > 0:
                flag.append(1)  ###### print(f'{name} has non-zero weights')
               
                continue
            else:
                flag.append(0)                
               
    layername = get_layer_name(layername)
    
    flag = np.asarray(flag)
    flag = np.expand_dims(flag,1)
    all_ = np.ones_like(flag)
    data = np.concatenate((all_, flag), axis =1)
    # print(data.shape)
    return data, layername


def check_zero_param_filterwise(model, config):
    
    layername = []
    _filters = []
    sparse_filters= []
    for name, param in model.named_parameters():
        
        if ('backbone' in name) and ('weight' in name) and ('conv' in name) and (len(param.shape) == 4):
            layername.append(name)
            # print(name, param.shape)
            count = 0
            for i in range(param.shape[1]):
                if torch.nonzero(param[:,i,:,:]).size(0) > 0:
                        ########non-zero values 
                    count += 1
                else:
                    continue
                    
                    
            sparse_filters.append(count)   
            _filters.append(param.shape[1])    
                    
    assert len(sparse_filters) == len(_filters)
    assert len(sparse_filters) == len(layername)
    layername = get_layer_name(layername)
    
    return layername, _filters, sparse_filters


#### calculate percentage sparsity
def calculate_percentage_sparsity(model):
    num_layer = 0
    num_nonzero = 0
    num_zero = 0
    total_params = 0
    non_zero_params = 0
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('weight' in name) and (len(param.shape) == 4):
            
            for i in range(param.shape[1]):
                num_layer += 1
                total_params += len(param[:,i,:,:].flatten())
                
                if torch.nonzero(param[:,i,:,:]).size(0) > 0:
                    num_nonzero += 1
                    non_zero_params += len(param[:,i,:,:].flatten())
                else:
                    num_zero += 1
                    
    assert num_layer == (num_nonzero + num_zero)    
    spar = (num_zero / num_layer)*100    
    
    sparsity_params = ((total_params - non_zero_params)/total_params)*100
    return spar, sparsity_params, total_params, non_zero_params ##### use non_zero for plots


def plot_sparsity(df, data, lname, gp_spar, dir, config):
    # print(df)
    f,axs = plt.subplots(2, figsize =(30,20),gridspec_kw = {'wspace':0, 'hspace':0}) ### 2
    f.tight_layout()
    # plotting columns ### 45,30
    xa = sns.barplot(x=df["layername"], y=df["filter_length"],color='royalblue',ax=axs[0])
    # for i in xa.containers:
    #     xa.bar_label(i,label_type = 'edge', color = 'royalblue')
    axx = sns.barplot(x=df["layername"], y=df["sparse_filters"], color='lightsalmon',ax=axs[0])
    # print(axx)
    # for i in axx.containers:        
    #     axx.bar_label(i,label_type = 'center')
    # axs[0].legend(loc="upper left", frameon=True, fontsize=20)
    axs[0].tick_params(axis='x', rotation=90, labelsize=18)
    # renaming the axes
    # axs[0].set(xlabel="layer names", ylabel="No. of filters")
    axs[0].set(xlabel=" ", ylabel=" ")
    

    # Create proxy artists for the legend
    no_sparsity_patch = mpatches.Patch(color='royalblue', label='total number of groups for each layer')
    sparse_patch = mpatches.Patch(color='lightsalmon', label='sparse groups, no. of non-zero groups')

    # Add legend to the first plot
    axs[0].legend(handles=[no_sparsity_patch, sparse_patch], loc="upper left", frameon=True, fontsize=40)
    
    axs[0].annotate(f'Group Sparsity: {gp_spar:.2f}%',xy=(0.15, 0.7), xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white'),
            fontsize=40, color='black',ha='center', va='center')

    axs[0].annotate(config['dataset_name'] + ' dataset',
                    xy=(0.08, 0.65), xycoords='axes fraction', 
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white'),
                    fontsize=40, color='black',
                    ha='center', va='top')
    
    task_names = ' + '.join(config['task_list'])
    axs[0].annotate(config['setup']+': '+ task_names ,
                    xy=(0.35, 0.58), xycoords='axes fraction', 
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='white', facecolor='white'),
                    fontsize=40, color='black',
                    ha='center', va='top')

    axs[1].spy(np.transpose(data), markersize=20)
    axs[1].set_xticks(np.arange(len(data)), labels = lname, rotation = 90, size = 18)
    axs[1].set_yticks(np.arange(2),labels=['ResNet50', 'sparse'])
    fname = dir + 'sparsity_plot.png'
    plt.savefig(fname, dpi=400)
    
    wandb.log({"Sparsity_plots": wandb.Image(f)})


   


def main():
    
    dir_checkpoint = args.exp_folder_path

    # dir_checkpoint = "/home/ricupa/Documents/MTL_meta_adaptive_features/results/runs/1_seg_trial_1/"
    last_element = os.path.basename(os.path.normpath(dir_checkpoint))
    config = create_config(dir_checkpoint +'config_file.yaml') 
    
    
    fname = config['checkpoint_folder'] + config['Experiment_name']
    runs = wandb.init(project= 'MTL_sparsity_3_outputs',name=last_element, entity='ricupa', config=config, dir = fname)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'    

    # if config['setup'] == 'singletask':
    #     print('singletask')
    model = get_model(config)
    model = model.to(device)

    checkpoint = torch.load(dir_checkpoint + '/checkpoint.pt')
    # model =   
    model.load_state_dict(checkpoint['model'])     
    data, lname = check_zero_params(model, config)
    spar, sparsity_params, total_params, non_zero_params = calculate_percentage_sparsity(model)
    total_params_str = f'{total_params / 1e6:.2f}M'
    non_zero_params_str = f'{non_zero_params / 1e6:.2f}M'
    print(f'sparsity_percentage: {spar:.2f}, parameter sparsity: {sparsity_params:.2f}, non_zero_params: {non_zero_params_str}, total_params: {total_params_str}')    
    layername, _filters, sparse_filters = check_zero_param_filterwise(model, config)
    df = pd.DataFrame({'layername': layername, 'filter_length': _filters, 'sparse_filters':sparse_filters})

    plot_sparsity(df, data, lname, spar, dir_checkpoint,config)        

     

    # else:
        
    #     for i, task in enumerate(config['task_list']):      
    #         task_chkpt = dir_checkpoint + task+'_checkpoint.pt'        
    #         if os.path.exists(task_chkpt): 
    #             print('task checkpoint exist :', task)   
    #             model = get_model(config)
    #             model = model.to(device)
    #             checkpoint = torch.load(task_chkpt)
    #             model.load_state_dict(checkpoint['model'])
    #             # print('for '+ task)
    #             _data, _layers = check_zero_params(model, config)
    #             layername, _filters, sparse_filters = check_zero_param_filterwise(model, config)
                
    #             spar, sparsity_params, total_params, non_zero_params = calculate_percentage_sparsity(model)
    #             total_params_str = f'{total_params / 1e6:.2f}M'
    #             non_zero_params_str = f'{non_zero_params / 1e6:.2f}M'
    #             print(f'sparsity_percentage: {spar:.2f}, parameter sparsity: {sparsity_params:.2f}, non_zero_params: {non_zero_params_str}, total_params: {total_params_str}')
    #             # flops, params = flopth(model, in_size=((3, 256, 256),),show_detail=True)
    #             # print(flops)
     
                
    #         else:
    #             print('task checkpoint not exist :', task) 
    #             model = get_model(config)
    #             model = model.cuda()
    #             checkpoint = torch.load(dir_checkpoint + 'checkpoint.pt')
    #             model.load_state_dict(checkpoint['model'])
    #             # print('for '+ task)
    #             _data, _layers = check_zero_params(model, config)
    #             layername, _filters, sparse_filters = check_zero_param_filterwise(model, config)
                
    #             spar, sparsity_params, total_params, non_zero_params = calculate_percentage_sparsity(model)
    #             total_params_str = f'{total_params / 1e6:.2f}M'
    #             non_zero_params_str = f'{non_zero_params / 1e6:.2f}M'
    #             print(f'sparsity_percentage: {spar:.2f}, parameter sparsity: {sparsity_params:.2f}, non_zero_params: {non_zero_params_str}, total_params: {total_params_str}')
                
    #             # flops, params = flopth(model, in_size=((3, 256, 256),),show_detail=True)
    #             # print(flops)

                
    #         if i == 0:
    #             param_data = _data
    #             no_sparse = ['No_sparse']*len(_filters)
    #             task_name = [task]*len(_filters)
    #             df = pd.DataFrame({'layername': layername, 'sparse_filters': sparse_filters, 'task': task_name})
    #             df_temp = pd.DataFrame({'layername': layername, 'sparse_filters': _filters, 'task': no_sparse})
    #             df = df.append(df_temp,ignore_index = True)
                
    #         else:        
    #             param_data = np.column_stack((param_data,np.expand_dims(_data[:,-1],1)))
    #             task_name = [task]*len(_filters)
    #             df_temp = pd.DataFrame({'layername': layername, 'sparse_filters': sparse_filters, 'task': task_name})
    #             df = df.append(df_temp,ignore_index = True)
            

    #     f,axs = plt.subplots(2, figsize =(45,30),gridspec_kw = {'wspace':0, 'hspace':0})
            
    #     # plotting columns
    #     sns.barplot(x='layername', y='sparse_filters', hue='task', data=df, ax=axs[0], palette= 'viridis') ###
    #     axs[0].tick_params(axis='x', rotation=90, labelsize=18)
    #     # renaming the axes
    #     # axs[0].set(xlabel="layer names", ylabel="No. of filters")
    #     axs[0].set(xlabel=" ", ylabel=" ")
    #     axs[0].legend(loc="upper left", frameon=True, fontsize=20)


    #     axs[1].spy(np.transpose(param_data), markersize=20)
    #     axs[1].set_xticks(np.arange(len(param_data)), labels = _layers, rotation = 90, size = 18)

    #     labels = [] 
    #     labels.append('ResNet50')
    #     for task in config['task_list']:
    #         labels.append(task)
    #     axs[1].set_yticks(np.arange(len(config['task_list'])+1),labels=labels)
        
    #     name = dir_checkpoint +'sparsity_plot.png'
    #     plt.savefig(name, dpi=400)
        
    #     wandb.log({"Sparsity_plots": wandb.Image(f)})
    
        
        
        

if __name__ == "__main__":
    main()
    
