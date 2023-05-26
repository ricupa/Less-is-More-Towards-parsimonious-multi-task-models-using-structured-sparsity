
##### conda activate MTLenv
###### python get_out_put_celeb.py --exp_name 8_2_multi_seg_male_smile_ --num_trials 5




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
from utils.utils_common import *
from train_val_test.trainer_class import Trainer
import json
import collections
import argparse
from collections import defaultdict, abc
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--exp_name', help='experiment or folder name of the model that needs to be evaluated, if there are several trials remember to check the experiment name in the main function')
parser.add_argument('--num_trials', help='if the training was done for multiple trials enter the number of trials otherwise 1')
args = parser.parse_args()


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_dict_to_csv(all_avg_dict, output_file):
    flattened_dict = flatten_dict(all_avg_dict)
    df = pd.DataFrame(list(flattened_dict.items()), columns=['key', 'mean_std'])
    # df.to_csv(output_file, index=False)
    df_transposed = df.set_index('key').T
    df_transposed.to_csv(output_file, index=False)



def mean_std(data):
    mean = np.mean(data)
    std = np.std(data)
    result = f"{mean:.4f} ± {std:.4f}"
    return result

def calculate_percentage_sparsity(model):
    num_layer = 0
    num_nonzero = 0
    num_zero = 0
    total_params = 0
    non_zero_params = 0
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('weight' in name) and (len(param.shape) == 4):
            
            for i in range(param.shape[0]):
                num_layer += 1
                total_params += len(param[i,:,:,:].flatten())
                
                if torch.nonzero(param[i,:,:,:]).size(0) > 0:
                    num_nonzero += 1
                    non_zero_params += len(param[i,:,:,:].flatten())
                else:
                    num_zero += 1
                    
    assert num_layer == (num_nonzero + num_zero)    
    spar = (num_zero / num_layer)*100    
    
    sparsity_params = ((total_params - non_zero_params)/total_params)*100
    return spar, sparsity_params, total_params, non_zero_params


def main():
    
    root_dir = "../results/runs/"
    
    exp = args.exp_name   

    num_trials = args.num_trials
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_metric_dict = collections.defaultdict(lambda: defaultdict(list))
    
    for i in range(num_trials):

        print('for trial -', i)
        
        if num_trials != 1:
            Exp_name = exp+'trial_'+str(i)
        else:
            Exp_name = exp

        dir_checkpoint = root_dir + Exp_name
        config = create_config(dir_checkpoint +'/config_file.yaml') 
        config['wandb_img_log'] = False
        model = get_model(config)
        model = model.to(device)
        
        trainer = Trainer(config, model, device)  
        epoch = 0

        if config['setup'] == 'singletask':
            print('singletask')
            checkpoint = torch.load(dir_checkpoint + '/checkpoint.pt')
            model.load_state_dict(checkpoint['model'])
            test_loss, test_metric = trainer.test(epoch, model)
            # print('avg_test_metric:', test_metric)
            spar, sparsity_params, _ ,_ = calculate_percentage_sparsity(model)
        else:
            test_metric = {}
            checkpoint = torch.load(dir_checkpoint + '/checkpoint.pt')
            model.load_state_dict(checkpoint['model'])
            
            for task in config['task_list']:
                task_chkpt = dir_checkpoint + '/' + task + '_checkpoint.pt'
                if os.path.exists(task_chkpt):
                    print('task checkpoint exists')
                    checkpoint = torch.load(task_chkpt)
                    model.load_state_dict(checkpoint['model'])
                else:
                    print('task checkpoint not found')
                
                loss, metric = trainer.test(epoch, model)
                test_metric[task] = metric[task]
                spar, sparsity_params, _ ,_ = calculate_percentage_sparsity(model)
            
                # print('avg_test_metric:', test_metric)

        
        test_metric['gp_spar'] = spar
        test_metric['param_spar'] = sparsity_params
        
        for key, value in test_metric.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():                        
                    all_metric_dict[key][sub_key].append(float(sub_value))
            else:
                all_metric_dict[key]["value"].append(float(value))


    
    all_avg_dict = collections.defaultdict(lambda: collections.defaultdict(list)) 
    
    for key,value in all_metric_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items(): 
                print(key, sub_key, mean_std(sub_value))
                all_avg_dict[key][sub_key] = mean_std(sub_value)
        else:
            print(key, mean_std(value))
            all_avg_dict[key] = mean_std(value)
        
    print('overall :', all_avg_dict)



    
    save_dict_to_csv(all_avg_dict, 'out_csv/' + exp + '.csv')



if __name__ == "__main__":
    main()
