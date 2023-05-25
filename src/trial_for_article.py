
##### conda activate MTLenv
##### python article.py



import os
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"]= '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
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
import seaborn as sns
from dataset.create_dataset import *


def get_layer_name(lname):
    layer_name = []
    for i, name in enumerate(lname):
        temp = name.split('.')
        temp1 = temp[1:-1]
        n = '.'.join(temp1)
        layer_name.append(n)
    return layer_name

def check_zero_param_filterwise(model, config):
    
    layername = []
    _filters = []
    sparse_filters= []
    for name, param in model.named_parameters():
        
        if ('backbone' in name) and ('weight' in name) and ('conv' in name) and (len(param.shape) == 4):
            layername.append(name)
            # print(name, param.shape)
            count = 0
            for i in range(param.shape[0]):
                if torch.nonzero(param[i,:,:,:]).size(0) > 0:
                        ########non-zero values 
                    count += 1
                else:
                    continue
                    
                    
            sparse_filters.append(count)   
            _filters.append(param.shape[0])    
                    
    assert len(sparse_filters) == len(_filters)
    assert len(sparse_filters) == len(layername)
    layername = get_layer_name(layername)
    
    return layername, _filters, sparse_filters

def check_zero_params(model, config):
    flag = []
    layername = []
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('weight' in name) and ('conv' in name) and (len(param.shape) == 4):
            layername.append(name)
            if torch.nonzero(param).size(0) > 0:
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





def main():
    list_exp = ['8_multi_seg_sn_depth_1e-5_trial_2', '7_multi_sn_depth_1e-5_trial_2', '1_single_seg_1e-5_trial_3','3_single_sn_1e-5_trial_4'] ####,'2_single_depth_1e-5_trial_3'
    cp_folder = '/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/new/'

    names = ['seg + sn + depth', 'sn + depth', 'seg', 'sn', 'depth']
    for i, exp in enumerate(list_exp):
        config = create_config(cp_folder+ exp +'/config_file.yaml') 
        
        model = get_model(config)   
        # model = model.cuda() 
        checkpoint = torch.load(cp_folder + exp  + '/checkpoint.pt') 
        model.load_state_dict(checkpoint['model'])     
        _data, _layers = check_zero_params(model, config)
        layername, _filters, sparse_filters = check_zero_param_filterwise(model, config)
        
        if i == 0:
            param_data = _data
            no_sparse = ['No_sparse']*len(_filters)
            task_name = [names[i]]*len(_filters)
            df = pd.DataFrame({'layername': layername, 'sparse_filters': sparse_filters, 'task': task_name})
            df_temp = pd.DataFrame({'layername': layername, 'sparse_filters': _filters, 'task': no_sparse})
            df = df.append(df_temp,ignore_index = True)
            
        else:        
            param_data = np.column_stack((param_data,np.expand_dims(_data[:,-1],1)))
            task_name = [names[i]]*len(_filters)
            df_temp = pd.DataFrame({'layername': layername, 'sparse_filters': sparse_filters, 'task': task_name})
            df = df.append(df_temp,ignore_index = True)


    # print(df)
    
    
    
    f,axs = plt.subplots(2, figsize =(50,35),gridspec_kw = {'wspace':0, 'hspace':0})
    f.tight_layout()       
    # plotting columns
    sns.barplot(x='layername', y='sparse_filters', hue='task', data=df, ax=axs[0], palette= 'tab10') ###
    axs[0].tick_params(axis='x', rotation=90, labelsize=18)
    # renaming the axes
    # axs[0].set(xlabel="layer names", ylabel="No. of filters")
    axs[0].set(xlabel=" ", ylabel=" ")
    axs[0].legend(loc="upper left", frameon=True, fontsize=20)


    axs[1].spy(np.transpose(param_data), markersize=20)
    axs[1].set_xticks(np.arange(len(param_data)), labels = _layers, rotation = 90, size = 18)

    labels = [] 
    labels.append('ResNet50')
    for task in config['task_list']:
        labels.append(task)
    axs[1].set_yticks(np.arange(len(config['task_list'])+1),labels=labels)

    plt.savefig('all_exp.png', dpi=400)







if __name__ == "__main__":
    main()
    