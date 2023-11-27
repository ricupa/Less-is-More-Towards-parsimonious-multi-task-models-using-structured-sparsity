# Less is More-- Towards parsimonious multi-task models using structured sparsity

Model sparsification in deep learning promotes simpler, more interpretable models with fewer parameters. This not only reduces the model's memory footprint and computational needs but also shortens inference time. This work focuses on creating sparse models optimized for multiple tasks with fewer parameters. These parsimonious models also possess the potential to match or outperform dense models in terms of performance. In this work, we introduce channel-wise l1/l2 group sparsity in the shared convolutional layers parameters (or weights) of the multi-task learning model. This approach facilitates the removal of extraneous groups i.e., channels (due to l1 regularization) and also imposes a penalty on the weights, further enhancing the learning efficiency for all tasks (due to l2 regularization). We analyzed the results of group sparsity in both single-task and multi-task settings on two widely-used MTL datasets: NYU-v2 and CelebAMask-HQ. On both datasets, which consist of three different computer vision tasks each, multi-task models with approximately 70\% sparsity outperform their dense equivalents. We also investigate how changing the degree of sparsification influences the model's performance, the overall sparsity percentage, the patterns of sparsity, and the inference time.

### Conda enviornment
Install the required python packages in a conda enviornment using the requirement.yml file, by using the following command:

conda env create -f requirement.yml

Note: wandb is used for logging the results of all the experiments. Remember to enter the correct entity name (yours) and experiment name in the main files before starting to train the models. 

### Dataset
In this work two publicly available datasets are used -
* NYU-v2 dataset -- the formatted multi-task version of the dataset can be downloaded from [link](https://drive.google.com/file/d/11pWuQXMFBNMIIB4VYMzi9RPE-nMOBU8g/view)
* CelebAMask-HQ dataset -- the formatted version can be found [here](https://drive.google.com/drive/folders/1bEn1gQO4GoIXI-VH8heESyFtOF6HUGhh?usp=share_link).
  
The actual CelebAMask-HQ dataset can be found [here](https://github.com/switchablenorms/CelebAMask-HQ), for the semantic segmentation task in this work the dataset was formatted for only 3 segmentation classes. The classification tasks are extracted from the attribute annotations. 


### Experiment configuration files

For analysing the performance of the proposed method which includes introducing group sparsity in multi-task learning setting, a wide range of experiments were conducted.
Using the config files both single task and multi-task experiments (with and without sparsity) can be designed for both NYU and celebA dataset.

* For NYU dataset - config/exp_single_mtl_NYU.yaml, 
* For celebA dataset - config/exp_single_multi_celebA.yaml
  
Remember to change the setup (singletask or multitask), group_sparsity (True or False) accordingly. 
The value of regularization parameter (sparsity strength) lambda can be changed as required under the 'BACKBONE OPTIMIZER PARAMETERS' i.e., {bb_optimizer_params}{lambda} 
The num_trials is the number of times you want to train each model. 
In this work every task is trained 5 times. 
Use the config files as required for changing the learning rates, task names, batch sizes, etc. 


### Training
After setting all the parameters as required in the config file, for training the models use the main_multi_trials.py files. 
For single task and multi-task training --

python main_multi_trials.py --config_exp config/exp_single_mtl_NYU.yaml  (for NYU-v2 dataset)

python main_multi_trials.py --config_exp config/exp_single_multi_celebA.yaml (for celebA dataset)


### Inference 

Few of the trained models can be found [here](https://drive.google.com/drive/folders/1bEn1gQO4GoIXI-VH8heESyFtOF6HUGhh?usp=share_link). As is it very diffucult to share all the trained models only are few of them are shared for reproducibility, rest of the models can be requested (from the suthors) when required.

To evaluate the models --

For celebA dataset : python get_out_put_celeb.py --exp_name 8_2_multi_seg_male_smile_ --num_trials 5

For NYU dataset : python get_outputs.py --exp_name 8_2_multi_seg_sn_depth_ --num_trials 5

After exp_name add the name of the experiment or the folder where the model is saved. 
Note - remember to change the path in the main function of the files. 


All the results are saved in a csv file in the out_csv folder

### Sparsity plots

To plot the sparsity plots of any trained model -

python plot_sparsity_wandb.py --exp_folder_path give_path_to_the folder_where_model_is_saved


### Group sparsity
The concept of group sparsity is included in the optimizer (ADAM), this is adapted (and modified accordingly) from the article ["Structured Sparsity Inducing Adaptive Optimizers for Deep Learning Tristan"](https://arxiv.org/abs/2102.03869).
The git of this work can be found [here](https://github.com/tristandeleu/pytorch-structured-sparsity)
