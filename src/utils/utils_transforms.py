import numpy.random as random
import numpy as np
import torch

import math
# import utils.helpers as helpers
import torchvision
import torchvision.transforms.functional as TF
# import torchvision.transforms.InterpolationMode
# from albumentations.pytorch import ToTensorV2
# from torchvision import transforms

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):
        for elem, val in sample.items():   

            if elem == 'image':                
                sample[elem] = self.to_tensor(val) 

            else:
                for tasks in sample[elem].keys(): 
                    if (tasks != 'segmentsemantic'):                    ##### &(tasks != 'edge_texture')
                        sample[elem][tasks] = self.to_tensor(sample[elem][tasks])

        return sample

    def __str__(self):
        return 'ToTensor'



# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std
#         self.normalize = torchvision.transforms.Normalize(self.mean, self.std)

#     def __call__(self, sample):
#         sample['image'] = self.normalize(sample['image'])    
#         return sample

#     def __str__(self):
#         return 'Normalize([%.3f,%.3f,%.3f],[%.3f,%.3f,%.3f])' %(self.mean[0], self.mean[1], self.mean[2], self.std[0], self.std[1], self.std[2])


class FixedResize(object):

    def __init__(self, dim):
        self.dim = dim
        self.resize_1 = torchvision.transforms.Resize(self.dim, interpolation= torchvision.transforms.InterpolationMode.NEAREST)
        self.resize_2 = torchvision.transforms.Resize(self.dim)

    def __call__(self, sample):
        for elem in sample.keys():            
            if elem == ('image'):                
                sample[elem] = self.resize_2(sample[elem])
            else:
                for tasks in sample[elem].keys():          
                        
                    if (tasks == 'segmentsemantic'): 
                        sample[elem][tasks] = self.resize_1(sample[elem][tasks])
                    
                    elif (tasks == 'depth_euclidean') or (tasks == 'surface_normal') or (tasks == 'edge_texture'):
                        sample[elem][tasks] = self.resize_2(sample[elem][tasks])
                    else:
                        continue 

        return sample

    def __str__(self):
        return 'FixedResize: '+str(self.dim)
    

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):
        if random.random() < 0.5:
            for elem, val in sample.items():
                
                if elem == 'image':
                    tmp = sample[elem]
                    tmp = torchvision.transforms.functional.hflip(tmp)
                    sample[elem] = tmp

                
                else:
                    for tasks in sample[elem].keys():                        
                        if (tasks == 'segmentsemantic') or (tasks == 'depth_euclidean') or (tasks == 'surface_normal')or (tasks == 'edge_texture'):
                            tmp = sample[elem][tasks]                            
                            tmp = torchvision.transforms.functional.hflip(tmp)
                            sample[elem][tasks] = tmp   
                        else:
                            continue
                                                
                     

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class RandomRotate(object):
    """Rotate the given image and ground truth randomly with a given angle."""
    def __init__(self, degrees):
        self.degrees = degrees
        

    def __call__(self, sample):
        if random.random() < 0.5:
            for elem, val in sample.items():
                
                if elem == 'image':
                    tmp = sample[elem]
                    tmp = torchvision.transforms.functional.rotate(tmp, self.degrees)
                    sample[elem] = tmp

                
                else:
                    for tasks in sample[elem].keys():                        
                        if (tasks == 'segmentsemantic') or (tasks == 'depth_euclidean') or (tasks == 'surface_normal') or (tasks == 'edge_texture'):
                            tmp = sample[elem][tasks]                            
                            tmp = torchvision.transforms.functional.rotate(tmp, self.degrees)
                            sample[elem][tasks] = tmp   
                        else:
                            continue
                                                
                     

        return sample

    def __str__(self):
        return 'RandomRotate'









