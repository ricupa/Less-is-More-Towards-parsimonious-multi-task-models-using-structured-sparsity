import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, decoders: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoders = decoders 
        self.task = task

    def forward(self, x):
        # out_size = x.size()[2:]
        out = self.decoders(self.backbone(x))
        return {self.task: out}


class MultiTaskModel(nn.Module):
   
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x):
        # start_time = time.time()             
        # shared_representation = self.backbone(x)
        # end_time = time.time()
        # total_time = end_time - start_time
        # total_time = total_time*1000
        # print(f"Inference time: {total_time} ms")
        # shared_representation.requires_grad_()
        # print(shared_representation.shape)
        # out = {}
        # for task in self.tasks:
        #     out[task] = self.decoders[task](shared_representation)            
        # return out
        
        return {task: self.decoders[task](self.backbone(x)) for task in self.tasks}