import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision.io import read_image
import PIL
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
# from convert16_to_8_bit import map_uint16_to_uint8
# import cv2
import json

def normalize_16bit(resized):
    pixels = np.asarray(resized)
    # convert from integers to floats
    pixels = pixels.astype('float32')
    # calculate global mean and standard deviation
    mean, std = pixels.mean(), pixels.std()
    #print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pixels = (pixels - mean) / std
    # clip pixel values to [-1,1]
    pixels = np.clip(pixels, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels = (pixels + 1.0) / 2.0
    # confirm it had the desired effect
    mean, std = pixels.mean(), pixels.std()
    return pixels,mean,std


class NYUDataset(Dataset):
    def __init__(self, config, data_dir, set, transform):
        self.data_dir = data_dir   
        if 'meta' in config.keys():
            self.splits_dir = os.path.join(self.data_dir, 'meta_gt_sets')  ###meta_
        else:
            self.splits_dir = os.path.join(self.data_dir, 'gt_sets') 
        self.set = set
        with open(self.splits_dir+'/'+ self.set+'.txt', 'r') as f:
            self.lines = f.read().splitlines()
        self.transform = transform
        self.task_list = config['task_list']     


    def __len__(self):      
        return len(self.lines) 
        # return 10       


    def __getitem__(self, idx): 
        
        sample= {}            
        image_name = self.data_dir +'/'+ 'images' +'/'+ self.lines[idx] +'.jpg'
        
       
        sample['image'] = Image.open(image_name).convert('RGB')
        

        sample['targets']={}
        for task in self.task_list:
            if task == 'segmentsemantic':
                label_name = self.data_dir +'/'+ 'segmentation' +'/'+ self.lines[idx] +'.png'
                img = read_image(label_name)  
                img = img -1
                img[img == -1] = 255              
                sample['targets']['segmentsemantic'] = img


            elif task == 'depth_euclidean':
                label_name = self.data_dir +'/'+ 'depth' +'/'+ self.lines[idx] +'.npy'
                depth_map = np.load(label_name, allow_pickle= True)   
                depth_map = depth_map / depth_map.max()
                max_depth = min(300, np.percentile(depth_map, 99))
                depth_map = np.clip(depth_map, 0.1, max_depth)             
                # depth_img = depth_img/depth_img.max()             
                sample['targets']['depth_euclidean'] = depth_map

            elif task == 'surface_normal':
                label_name = self.data_dir +'/'+ 'normals' +'/'+ self.lines[idx] +'.npy'
                sn_img = np.load(label_name, allow_pickle= True) 
                # sn_mask = torch.zeros(sn_img.shape)
                # sn_mask[sn_img ==0] = 1  
                # sn_img = (sn_img - sn_img.min())/(sn_img.max() - sn_img.min())          
                sample['targets']['surface_normal'] = sn_img 
                # sample['targets']['surface_normal_mask'] = torch.tensor(sn_mask)

            elif task == 'edge_texture':
                label_name = self.data_dir +'/'+ 'edge' +'/'+ self.lines[idx] +'.npy'
                edge_img = np.load(label_name, allow_pickle= True)
                edge_img = edge_img - edge_img.min()/ (edge_img.max() - edge_img.min())    
                # edge_img = (edge_img>0.5)*1.0
                # edge_img = torch.tensor(edge_img)
                # edge_img = torch.unsqueeze(edge_img,0)
       

                sample['targets']['edge_texture'] = edge_img

            else:
                print('Task not found :', task)

        if self.transform:            
            sample = self.transform(sample)

        return sample

class CelebMaskHQDataset():
    def __init__(self,config,set,transform=None):
        self.config = config        
        self.img_path = os.path.join(config['data_dir_celebA'],'CelebA-HQ-img')
        self.seg_mask_path = os.path.join(config['data_dir_celebA'],'CelebA-HQ-segmask')
        with open('dataset/'+"celebA_datasplit.json", 'r') as f:
            data_split = json.load(f)
        self.data_list = data_split[set]        
        self.transform = transform
        ### load the file with all the 40 attributes
        ann_path = os.path.join(config['data_dir_celebA'],'CelebAMask-HQ-attribute-anno.txt')
        temp_file = pd.read_csv(ann_path, sep=" ")
        # extract only 4 attributes
        self.attributes_list = ['Arched_Eyebrows', 'Eyeglasses', 'Smiling', 'Male']
        self.attributes = temp_file[self.attributes_list]
        new_index = [name[0] for name in self.attributes.index]
        self.attributes.index = new_index
        self.attributes = self.attributes.replace(-1, 0)
        
    def __len__(self):
        return int(len(self.data_list)/3)
    
    def __getitem__(self,index):    
        sample = {}  
        # print(self.data_list[index]) 
        try:
            img = Image.open(os.path.join(self.img_path,self.data_list[index]))
        except OSError as e:
            print(f"Could not open image: {e}")
            
        # img_mask = np.load(os.path.join(self.label_path,os.listdir(self.label_path)[index]))
        if self.transform is not None:
            img = self.transform(img)    
                
        sample['image'] = img
        
        sample['targets'] = {}        
         
        
        for task in self.config['task_list']:  

            if task == 'segmentsemantic':
                name = "{:05d}_mask.npy".format(int(self.data_list[index][:-4]))     
                # print(name)           
                img_mask = np.load(os.path.join(self.seg_mask_path,name))
                
                if self.transform is not None:
                    img_mask = self.transform(img_mask)
                sample['targets']['segmentsemantic'] = img_mask

            elif task == 'class_eyebrows':        
                sample['targets']['class_eyebrows']= self.attributes.loc[self.data_list[index],'Arched_Eyebrows']
            
            elif task == 'class_glasses':
                sample['targets']['class_glasses']= self.attributes.loc[self.data_list[index],'Eyeglasses']
            elif task == 'class_smile':
                sample['targets']['class_smile']= self.attributes.loc[self.data_list[index],'Smiling']
            elif task == 'class_male':
                sample['targets']['class_male']= self.attributes.loc[self.data_list[index],'Male']
            else:
                print('invalid task name for celebA dataset')
        
        
        return sample 

