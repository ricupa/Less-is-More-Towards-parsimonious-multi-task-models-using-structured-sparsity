# Less-is-More-Group-Sparsity-for-Parsimonious-Multi-Task-Models


### Abstract

Group sparsity in Machine Learning (ML) encourages simpler, more interpretable models with fewer active parameter groups. This work aims to incorporate structured group sparsity into the shared parameters of a Multi-Task Learning (MTL) framework, to develop parsimonious models that can effectively address multiple tasks with fewer parameters while maintaining comparable or superior performance to a dense model. Sparsifying the model during training helps decrease the model’s memory footprint, computation requirements, and prediction time during inference. We use channel-wise l1/l2 group sparsity in the shared layers of the Convolutional Neural Network (CNN). This approach not only facilitates the elimination of extraneous groups (channels) but also imposes a penalty on the weights, thereby enhancing the learning of all tasks. We compare the outcomes of single-task and multi-task experiments under group sparsity on two publicly available MTL datasets, NYU-v2 and CelebAMask-HQ. We also investigate how changing the sparsification degree impacts both the performance of the model and the sparsity of groups.

### Dataset
In this work two publicly available datasets are used -
* NYU-v2 dataset -- the formatted multi-task version of the dataset can be downloaded from [link](https://drive.google.com/file/d/11pWuQXMFBNMIIB4VYMzi9RPE-nMOBU8g/view)
* ClebAMask-HQ dataset -- 


