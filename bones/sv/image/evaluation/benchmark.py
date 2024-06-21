import shap
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import time
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import copy
import os
from datetime import datetime
import subprocess
import torch.nn as nn

from ..explainers.vitshapley.vit_shapley.config import config, dataset_ImageNette, dataset_MURA, dataset_Pet
from ..explainers.vitshapley.vit_shapley.datamodules.ImageNette_datamodule import ImageNetteDataModule
# from ..explainers.vitshapley.vit_shapley.datamodules.MURA_datamodule import MURADataModule
from ..explainers.vitshapley.vit_shapley.datamodules.Pet_datamodule import PetDataModule
from ..explainers.vitshapley.vit_shapley.modules.explainer import Explainer
from ..explainers.vitshapley.vit_shapley.modules.surrogate import Surrogate
from ..explainers.vitshapley.vit_shapley.modules.classifier import Classifier

from torch.utils.data import Dataset, DataLoader
import torch

from .tabulate import tabulate
# print(os.getcwd())

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ClassifierPT(nn.Module):
    def __init__(self, backbone, head):
        super(ClassifierPT, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        # print(x)
        x = self.head(x["x"])
        return x

class ClassifierResPT(nn.Module):
    def __init__(self, backbone, head):
        super(ClassifierResPT, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x=torch.nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.backbone(x)
        # print(x)
        x = self.head(x["x"])
        return x


class Benchmark:
    def __init__(self, dataset, explainers, metrics, ground_truth, num_samples, classifier=None, sample_method='random'):
        if len(dataset) != 1:
            raise ValueError("Currently only one dataset is supported")
        
        self.dataset = dataset[0]()  # Currently is supported only one dataset at a time
        self.explainers = explainers
        self.explainers_init = {}
        self.metrics = metrics
        self.ground_truth = ground_truth
        self.ground_truth_name = {}
        self.num_samples = num_samples
        self.sample_method = sample_method
        self.kernelshap_iters = 128 # default value
        self.classifier = classifier

        self.config=config()
        if self.dataset.name == 'ImageNette':
            self.config.update(dataset_ImageNette())
        if self.dataset.name == 'Pet':
            self.config.update(dataset_Pet())

        self.config.update(
                {'gpus_surrogate':[0,],
                'gpus_explainer':[0,],
                'gpus_classifier':[0,]}
            )
        
        self.train_set=None
        self.val_set=None
        
    


    def run(self, verbose=False, load=False): # CURRENTLY THE GROUND TRUTH IS THE SAME FOR EACH DATASET
            
            # rc = subprocess.call(f'{os.getcwd()}/bones/sv/image/benchmark/{self.dataset.name}_CLF.sh' ,shell=True)

            _config = self.config

            # pl.seed_everything(seed=_config["seed"])

            # # _config["exp_name"] = datetime.now().strftime("%y%m%d_%H%M%S") + "_" + _config['exp_name']
            # _config["exp_name"] = _config['exp_name']+'_CLF'
            # _config["classifier_backbone_type"]="vit_tiny_patch16_224"
            # _config["classifier_download_weight"]=True
            # _config["classifier_load_path"]=None
            # _config['checkpoint_metric'] = 'accuracy'
            # _config['learning_rate'] = 1e-5

            # wandb_logger = set_wandb_logger(exp_name=_config["exp_name"],
            #                                 wandb_project_name=_config["wandb_project_name"],
            #                                 log_dir=_config["log_dir"],
            #                                 log_model=(_config["explanation_location_train"] is None))
            # wandb_logger.experiment.config.update(_config)
            
            dataset_parameters = {
                "dataset_location": _config["dataset_location"],
                "explanation_location_train": _config["explanation_location_train"],
                "explanation_mask_amount_train": _config["explanation_mask_amount_train"],
                "explanation_mask_ascending_train": _config["explanation_mask_ascending_train"],
                "explanation_location_val": _config["explanation_location_val"],
                "explanation_mask_amount_val": _config["explanation_mask_amount_val"],
                "explanation_mask_ascending_val": _config["explanation_mask_ascending_val"],
                "explanation_location_test": _config["explanation_location_test"],
                "explanation_mask_amount_test": _config["explanation_mask_amount_test"],
                "explanation_mask_ascending_test": _config["explanation_mask_ascending_test"],
                "transforms_train": _config["transforms_train"],
                "transforms_val": _config["transforms_val"],
                "transforms_test": _config["transforms_test"],
                "num_workers": _config["num_workers"],
                "per_gpu_batch_size": _config["per_gpu_batch_size"],
                "test_data_split": _config["test_data_split"],
            }

            if _config["datasets"] == "ImageNette":
                if verbose:
                    print("DATASET: ImageNette")
                datamodule = ImageNetteDataModule(**dataset_parameters)
                for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_classifier_BONES/checkpoints'):
                    if file.endswith(".ckpt"):
                        classifier_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_classifier_BONES/checkpoints/{file}'
                        break

                # classifier_backbone=f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_classifier_vit_tiny_patch16_224_1e-5_train_BONES/checkpoints/.ckpt'
            elif _config["datasets"] == "Pet":
                dataset_parameters["dataset_location"] = "./pets"
                datamodule = PetDataModule(**dataset_parameters)
                for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_classifier_BONES/checkpoints'):
                    if file.endswith(".ckpt"):
                        classifier_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_classifier_BONES/checkpoints/{file}'
                        break
            else:
                ValueError("Invalid 'datasets' configuration")

            # print(classifier_backbone)

            if os.path.isfile(classifier_backbone) and load:
                if verbose:
                    print("\nLOADING CLASSIFIER BACKBONE")
                classifier_VIT = Classifier(
                    backbone_type="vit_tiny_patch16_224",
                    download_weight=False,
                    load_path=classifier_backbone,
                    target_type=_config["target_type"],
                    output_dim=_config["output_dim"],
                    enable_pos_embed=_config["classifier_enable_pos_embed"],
                    checkpoint_metric=None,
                    optim_type=None,
                    learning_rate=None,
                    loss_weight=None,
                    weight_decay=None,
                    decay_power=None,
                    warmup_steps=None).to(_config["gpus_classifier"][0]
                )
                # print(classifier_VIT)
            else:
                if verbose:
                    print("\nTRAINING CLASSIFIER BACKBONE")
                rc = subprocess.call(f'{os.getcwd()}/bones/sv/image/benchmark/{self.dataset.name}_CLF.sh' ,shell=True)
                os.remove(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_classifier_BONES/checkpoints/last.ckpt')
                torch.cuda.empty_cache()
                if _config["datasets"] == "ImageNette":
                    print("DATASET: ImageNette")
                    datamodule = ImageNetteDataModule(**dataset_parameters)
                    for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_classifier_BONES/checkpoints'):
                        if file.endswith(".ckpt"):
                            classifier_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_classifier_BONES/checkpoints/{file}'
                            break

                    # classifier_backbone=f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_classifier_vit_tiny_patch16_224_1e-5_train_BONES/checkpoints/.ckpt'
                elif _config["datasets"] == "Pet":
                    dataset_parameters["dataset_location"] = "./pets"
                    datamodule = PetDataModule(**dataset_parameters)
                    for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_classifier_BONES/checkpoints'):
                        if file.endswith(".ckpt"):
                            classifier_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_classifier_BONES/checkpoints/{file}'
                            break
                else:
                    ValueError("Invalid 'datasets' configuration")
                
                classifier_VIT = Classifier(
                    backbone_type="vit_tiny_patch16_224",
                    download_weight=False,
                    load_path=classifier_backbone,
                    target_type=_config["target_type"],
                    output_dim=_config["output_dim"],
                    enable_pos_embed=_config["classifier_enable_pos_embed"],
                    checkpoint_metric=None,
                    optim_type=None,
                    learning_rate=None,
                    loss_weight=None,
                    weight_decay=None,
                    decay_power=None,
                    warmup_steps=None).to(_config["gpus_classifier"][0]
                )
            
            if verbose:
                print("\nInitialize Classifier")
            classifier_VIT.eval()
            backbone=classifier_VIT.backbone
            head=classifier_VIT.head
            backbone.eval()
            head.eval()
            device=classifier_VIT.device
            self.device=device

            classifierPT=ClassifierPT(backbone, head).to(device)
            classifierResPT=ClassifierResPT(backbone, head).to(device)
            self.classifier_VIT = classifier_VIT
            self.classifierPT = classifierPT
            self.classifierResPT = classifierResPT

            if verbose:
                print("\nInitialize Data")
            datamodule.set_test_dataset()
            datamodule.set_train_dataset()
            datamodule.set_val_dataset()
            train_dataset = datamodule.train_dataset
            val_dataset = datamodule.val_dataset

            if os.path.isfile(f'{os.getcwd()}/bones/sv/image/evaluation/train_data.pkl') and os.path.isfile(f'{os.getcwd()}/bones/sv/image/evaluation/val_data.pkl'):
                if verbose:
                    print("\tLoading train and val data")
                with open(f'{os.getcwd()}/bones/sv/image/evaluation/train_data.pkl', 'rb') as f:
                    train_data = pickle.load(f)
                with open(f'{os.getcwd()}/bones/sv/image/evaluation/val_data.pkl', 'rb') as f:
                    val_data = pickle.load(f)
            else:
                if verbose:
                    print("\tComputing train and val data")
                train_data=[(el["images"], int(el["labels"])) for el in train_dataset]
                val_data=[(el["images"], int(el["labels"])) for el in val_dataset]

                # save train and val data
                with open(f'{os.getcwd()}/bones/sv/image/evaluation/train_data.pkl', 'wb') as f:
                    pickle.dump(train_data, f)
                with open(f'{os.getcwd()}/bones/sv/image/evaluation/val_data.pkl', 'wb') as f:
                    pickle.dump(val_data, f)

            train_set=CustomDataset(train_data)
            val_set=CustomDataset(val_data)

            self.train_set = train_set
            self.val_set = val_set

            if verbose:
                print("\nInitialize Grand Truth")
            ground_model = self.ground_truth(self.dataset, self.config, classifierResPT, train_set, val_set, classifierPT, device)
            print("\nGround Truth:", ground_model.name)
            self.ground_truth_name[self.dataset.name]=ground_model.name

            # initialize explainers as a dictionary name:explainer
            explainers = {}
            explainers[ground_model.name] = ground_model
            if verbose:
                print("\nInitializing Explainers")
            for explainer in self.explainers:
                print("\tExplainer:", explainer) #still not initialized
                exp = explainer(self.dataset, self.config, classifierResPT, train_set, val_set, classifierPT, device, load, verbose)
                explainers[exp.name]=exp
            
            if verbose:
                print("\nExplainers:", explainers.keys())

            self.explainers_init[self.dataset.name] = explainers

            # initialize metrics as a dictionary name:metric
            metrics = {}
            for metric in self.metrics:
                mtr=metric()
                metrics[mtr.name]=mtr

            print("Metrics:", metrics.keys())


            if self.num_samples>len(train_set):
                if self.sample_method == "random":
                    # sample random samples from the dataset
                    indexes=[]
                    for i in range(self.num_samples):
                        indexes.append(random.sample(list(np.arange(len(train_set))), 1)[0])
                        
                    DATA=[]
                    LABELS=[]
                    for idx in indexes:
                        DATA.append(train_set[idx])
                    DATA=np.array(DATA)
                else:
                    print("Error: Number of samples is greater than the number of samples in the dataset")
                    print("Please set sample_method to 'random' or reduce the number of samples")
                    exit(1)
            else:
                DATA=train_set

            
            print("\nRunning Explainers")
            for IDX in tqdm(range(len(DATA[:self.num_samples]))):
                sample=DATA[IDX][0]
                label=DATA[IDX][1]
                # print(sample.shape)
                # print(label)
                # print(sample, label)

                for k, expl in explainers.items():
                    time_start = time.time()
                    out=expl.compute(sample, label)
                    time_end = time.time()
                    expl.list_time.append(time_end-time_start)
                    expl.list_sv.append(out[0])
                

                ground_sv=ground_model.list_sv[-1]
                for k, mtr in metrics.items():
                    for k, expl in explainers.items():
                        if expl.name != ground_model.name:
                            sv_comparison=expl.list_sv[-1]
                            res=mtr.compute(sv_comparison, ground_sv)
                            if mtr.name == 'L1':
                                expl.list_l1.append(res)
                            if mtr.name == 'L2':
                                expl.list_l2.append(res)
            
            if verbose:
                print("-"*100)
            

    def print_results(self, dsfn):
        explainers=self.explainers_init[dsfn().name]
        table = [['Method', 'Time Training', 'Time Inference', 'Time Total', 'L1', 'L2']]
        for k, expl in explainers.items():
            if len(expl.list_l1)==0:
                table.append([expl.name, expl.training_time, np.mean(expl.list_time), np.mean(expl.list_time)+expl.training_time, 0, 0])
            else:
                table.append([expl.name, expl.training_time, np.mean(expl.list_time), np.mean(expl.list_time)+expl.training_time, np.mean(expl.list_l1), np.mean(expl.list_l2)])


        print(tabulate(table, headers='firstrow', headersglobalalign='center', tablefmt='fancy_grid', colalign=('center','center','global','global','global','global','global')))

'''
'''