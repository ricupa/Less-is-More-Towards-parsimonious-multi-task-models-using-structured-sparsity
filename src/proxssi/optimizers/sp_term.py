# from math import sqrt
# from typing import Callable, Iterable, Optional, Tuple, Union

# import torch
# from proxssi.l1_l2 import prox_l1_l2_weighted
# from proxssi.mcp_l2 import prox_group_mcp_weighted
# from proxssi.groups.resnet_gp import resnet_groups
# from torch.nn.modules.module import Module


# class group_sparsity_term(Module):   

#     def __init__(self, config):
        
#         super().__init__()   
        
#         self.args = {'weight_decay': config['bb_optimizer_params']['weight_decay'],
#                 'learning_rate': config['bb_optimizer_params']['learning_rate']}        
        
#         self.config = config
#         self.penalty = config['optimizer_params']['penalty']
#         if self.penalty not in ['l1_l2', 'group_mcp']:
#             raise ValueError('Unknown penalty: {} - must be [l1_l2, group_mcp]'.format(self.penalty))
#         self.prox_kwargs = {'lr': config['bb_optimizer_params']['learning_rate'], 
#                             'lambda_': config['bb_optimizer_params']['lambda']}
        
        
#     def forward(self, model):        
#         sparsity_term = 0            
#         grouped_params = resnet_groups(model, self.args)        
#         if self.penalty == 'l1_l2':        
#             for group in grouped_params:                                                   
#                 for idx, p in enumerate(group["params"]):                       
#                     groups_fn = group['groups_fn'] 
#                     if groups_fn is not None:
#                         params_list = groups_fn(p)
#                         for params in params_list: 
#                             # print(params.shape)
#                             lambda_g = sqrt(params.numel()) * self.prox_kwargs['lambda_']
#                             params_ = params.reshape(-1)
#                             sparsity_term += torch.linalg.norm(params_, 2) * lambda_g
#                     # else:
#                     #     params_list = p
#                     #     lambda_g = sqrt(params.numel()) * self.prox_kwargs['lambda_']   
#                     #     params_ = params.reshape(-1)
#                     #     sparsity_term += torch.linalg.norm(params_, 2) * lambda_g 
                    
         
#         else:
#             print('mcp not imlemented yet')        
            
#         return sparsity_term
        
###################################################################

# from math import sqrt
# from typing import Callable, Iterable, Optional, Tuple, Union

# import torch
# from proxssi.l1_l2 import prox_l1_l2_weighted
# from proxssi.mcp_l2 import prox_group_mcp_weighted
# from proxssi.groups.resnet_gp import resnet_groups
# from torch.nn.modules.module import Module


# class GroupSparsityTerm(Module):

#     def __init__(self, config):
        
#         super().__init__()
        
#         self.args = {'weight_decay': config['bb_optimizer_params']['weight_decay'],
#                      'learning_rate': config['bb_optimizer_params']['learning_rate']}        
        
#         self.config = config
#         self.penalty = config['optimizer_params']['penalty']
#         if self.penalty not in ['l1_l2', 'group_mcp']:
#             raise ValueError('Unknown penalty: {} - must be [l1_l2, group_mcp]'.format(self.penalty))
#         self.prox_kwargs = {'lr': config['bb_optimizer_params']['learning_rate'], 
#                             'lambda_': config['bb_optimizer_params']['lambda']}
#         self.group_structure = None
#         self.lambda_g_values = None

#     def cache_group_structure(self, model):
#         self.group_structure = resnet_groups(model, self.args)
#         self.lambda_g_values = []
#         for group in self.group_structure:
#             group_lambda_g = []
#             for p in group["params"]:
#                 group_lambda_g.append(sqrt(p.numel()) * self.prox_kwargs['lambda_'])
#             self.lambda_g_values.append(group_lambda_g)

#     def forward(self, model):        
#         sparsity_term = 0
        
#         if self.group_structure is None:
#             self.cache_group_structure(model)

#         if self.penalty == 'l1_l2':
#             for group_idx, group in enumerate(self.group_structure):                                                   
#                 for idx, p in enumerate(group["params"]):                       
#                     groups_fn = group['groups_fn']
#                     if groups_fn is not None:
#                         params_list = groups_fn(p)
#                         for params_idx, params in enumerate(params_list):
#                             lambda_g = self.lambda_g_values[group_idx][idx]
#                             params_ = params.reshape(-1)
#                             sparsity_term += torch.linalg.norm(params_, 2) * lambda_g
#         else:
#             print('mcp not implemented yet')        
            
#         return sparsity_term
     
##############################################################

from math import sqrt
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from proxssi.l1_l2 import prox_l1_l2_weighted
from proxssi.mcp_l2 import prox_group_mcp_weighted
from proxssi.groups.resnet_gp import resnet_groups
from torch.nn.modules.module import Module
from proxssi.penalties import *

# class GroupSparsityTerm(Module):

#     def __init__(self, config, model):
#         super().__init__()

#         self.args = {'weight_decay': config['bb_optimizer_params']['weight_decay'],
#                      'learning_rate': config['bb_optimizer_params']['learning_rate']}

#         self.config = config
#         self.penalty = config['optimizer_params']['penalty']
#         if self.penalty not in ['l1_l2', 'group_mcp']:
#             raise ValueError('Unknown penalty: {} - must be [l1_l2, group_mcp]'.format(self.penalty))
#         self.prox_kwargs = {'lr': config['bb_optimizer_params']['learning_rate'],
#                             'lambda_': config['bb_optimizer_params']['lambda']}
#         self.group_structure = resnet_groups(model, self.args)
#         self.lambda_g_values = []
#         for group in self.group_structure:
#             group_lambda_g = []
#             for p in group["params"]:
#                 group_lambda_g.append(sqrt(p.numel()) * self.prox_kwargs['lambda_'])
#             self.lambda_g_values.append(group_lambda_g)

#     def forward(self) -> torch.Tensor:
#         sparsity_term = torch.tensor(0.0, device = self.config['device'])

#         if self.penalty == 'l1_l2':
#             for group_idx, group in enumerate(self.group_structure):
#                 for idx, p in enumerate(group["params"]):
#                     groups_fn = group['groups_fn']
#                     if groups_fn is not None:
#                         params_list = groups_fn(p)
#                         for params_idx, params in enumerate(params_list):
#                             lambda_g = self.lambda_g_values[group_idx][idx]
#                             params_ = params.reshape(-1)
#                             sparsity_term += torch.linalg.norm(params_, 2) * lambda_g
#         else:
#             print('mcp not implemented yet')

#         return sparsity_term
    
    

class GroupSparsityTerm(Module):
    
    def __init__(self, config, model):
        super().__init__()

        self.args = {'weight_decay': config['bb_optimizer_params']['weight_decay'],
                     'learning_rate': config['bb_optimizer_params']['learning_rate']}

        self.config = config
        self.penalty = config['optimizer_params']['penalty']
        if self.penalty not in ['l1_l2', 'group_mcp']:
            raise ValueError('Unknown penalty: {} - must be [l1_l2, group_mcp]'.format(self.penalty))
        self.prox_kwargs = {'lr': config['bb_optimizer_params']['learning_rate'],
                            'lambda_': config['bb_optimizer_params']['lambda']}
        self.group_structure = resnet_groups(model, self.args)
        self.lambda_g = config['bb_optimizer_params']['lambda']

    def forward(self) -> torch.Tensor:
        sparsity_term = torch.tensor(0.0, device = self.config['device'])

        if self.penalty == 'l1_l2':
            sparsity_term = l1_l2(self.group_structure, lambda_ = self.lambda_g )
        else:
            print('mcp not implemented yet')

        return sparsity_term




class Meta_GroupSparsityTerm(Module):
    
    def __init__(self, config, model, lambda_param):
        super().__init__()

        self.args = {'weight_decay': config['bb_optimizer_params']['weight_decay'],
                     'learning_rate': config['bb_optimizer_params']['learning_rate']}

        self.config = config
        self.penalty = config['optimizer_params']['penalty']
        if self.penalty not in ['l1_l2', 'group_mcp']:
            raise ValueError('Unknown penalty: {} - must be [l1_l2, group_mcp]'.format(self.penalty))
        self.prox_kwargs = {'lr': config['bb_optimizer_params']['learning_rate'],
                            'lambda_': lambda_param}
        self.group_structure = resnet_groups(model, self.args)
        self.lambda_g = lambda_param

    def forward(self) -> torch.Tensor:
        sparsity_term = torch.tensor(0.0, device = self.config['device'])

        if self.penalty == 'l1_l2':
            sparsity_term = l1_l2(self.group_structure, lambda_ = self.lambda_g )
        else:
            print('mcp not implemented yet')

        return sparsity_term