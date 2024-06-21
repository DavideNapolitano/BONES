
# from .shapley_regression import removal, s

import shap
import torch 
import torch.nn as nn
import time
import os
# print("models",os.getcwd())

from .vitshapley.vit_shapley.modules.explainer import Explainer
from .vitshapley.vit_shapley.modules.surrogate import Surrogate
from .vitshapley.vit_shapley.modules.classifier import Classifier

from .fastshap import image_surrogate
from .fastshap import fastshap
from .fastshap import resnet
from .fastshap import unet
from .fastshap.utils import DatasetInputOnly, MaskLayer2d, KLDivLoss

from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.ndimage import zoom
import sys
sys.path.append("bones/sv/image/explainers/vitshapley")
sys.path.append("..")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")


def aggregate_and_resize_shap_values(shap_values, new_size=(14, 14)):
    zoom_factor = (new_size[0] / shap_values.shape[0], new_size[1] / shap_values.shape[1])
    resized_shap_values = zoom(shap_values, zoom_factor, order=1)  
    return resized_shap_values

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class VitShapleyModel():
    def __init__(self, dataset, config, classifierResPT, train_set, val_set, classifierPT, device, load, verbose):
        self.name='VitShapley'
        self.training_time=0
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_sv=[]
        self.dataset = dataset
        self.config = config

        if self.dataset.name == 'ImageNette':
            for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_surrogate_BONES/checkpoints'):
                if file.endswith(".ckpt"):
                    surrogate_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_surrogate_BONES/checkpoints/{file}'
                    break
            for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_explainer_BONES/checkpoints'):
                if file.endswith(".ckpt"):
                    explainer_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_explainer_BONES/checkpoints/{file}'
                    break
        else:
            for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_surrogate_BONES/checkpoints'):
                if file.endswith(".ckpt"):
                    surrogate_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_surrogate_BONES/checkpoints/{file}'
                    break
            for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_explainer_BONES/checkpoints'):
                if file.endswith(".ckpt"):
                    surrogate_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_explainer_BONES/checkpoints/{file}'
                    break
        
        _config = self.config
        # surrogate model
        if os.path.isfile(surrogate_backbone) and load:
            if verbose:
                print("\t\tLOADING SURROGATE BACKBONE")
            # print(surrogate_backbone)
            # download_file(backbone_type_config["surrogate_path"], "surrogate.ckpt")
            surrogate_VIT = Surrogate(
                mask_location="pre-softmax",
                backbone_type="vit_tiny_patch16_224",
                download_weight=False,
                load_path=surrogate_backbone,
                target_type=_config["target_type"],
                output_dim=_config["output_dim"],

                target_model=None,
                checkpoint_metric=None,
                optim_type=None,
                learning_rate=None,
                weight_decay=None,
                decay_power=None,
                warmup_steps=None,
                load_path_state_dict=True
            ).to(_config["gpus_surrogate"][0])
        else:
            if verbose:
                print("\t\tTRAINING SURROGATE BACKBONE")
            time_start = time.time()
            rc = subprocess.call(f'{os.getcwd()}/bones/sv/image/benchmark/{self.dataset.name}_SUR.sh' ,shell=True)
            self.training_time += time.time()-time_start                
            os.remove(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_surrogate_BONES/checkpoints/last.ckpt')
            torch.cuda.empty_cache()
            if self.dataset.name == 'ImageNette':
                for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_surrogate_BONES/checkpoints'):
                    if file.endswith(".ckpt"):
                        surrogate_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_surrogate_BONES/checkpoints/{file}'
                        break
            else:
                for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_surrogate_BONES/checkpoints'):
                    if file.endswith(".ckpt"):
                        surrogate_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_surrogate_BONES/checkpoints/{file}'
                        break
            surrogate_VIT = Surrogate(
                mask_location="pre-softmax",
                backbone_type="vit_tiny_patch16_224",
                download_weight=False,
                load_path=surrogate_backbone,
                target_type=_config["target_type"],
                output_dim=_config["output_dim"],

                target_model=None,
                checkpoint_metric=None,
                optim_type=None,
                learning_rate=None,
                weight_decay=None,
                decay_power=None,
                warmup_steps=None,
                load_path_state_dict=True).to(_config["gpus_surrogate"][0])

        if os.path.isfile(explainer_backbone) and load:
            if verbose:
                print("\t\tLOADING EXPLAINER BACKBONE")
            # print(explainer_backbone)
            # download_file(backbone_type_config["explainer_path"], "explainer.ckpt")
            explainer_VIT = Explainer(
                normalization="additive",
                normalization_class=_config["explainer_normalization_class"],
                activation="tanh",
                surrogate=surrogate_VIT,
                link='sigmoid' if _config["output_dim"]==1 else 'softmax',
                backbone_type="vit_tiny_patch16_224",
                download_weight=False,
                residual=[],
                load_path=explainer_backbone,
                target_type=_config["target_type"],
                output_dim=_config["output_dim"],

                explainer_head_num_attention_blocks=1,
                explainer_head_include_cls=True,
                explainer_head_num_mlp_layers=3,
                explainer_head_mlp_layer_ratio=4,
                explainer_norm=_config["explainer_norm"],

                efficiency_lambda=_config["explainer_efficiency_lambda"],
                efficiency_class_lambda=_config["explainer_efficiency_class_lambda"],
                freeze_backbone="all",

                checkpoint_metric=_config["checkpoint_metric"],
                optim_type=_config["optim_type"],
                learning_rate=_config["learning_rate"],
                weight_decay=_config["weight_decay"],
                decay_power=_config["decay_power"],
                warmup_steps=_config["warmup_steps"],
                load_path_state_dict=True).to(_config["gpus_explainer"][0])   
        else:
            if verbose:
                print("TRAINING EXPLAINER BACKBONE")
            # explainer model
            time_start = time.time()
            rc = subprocess.call(f'{os.getcwd()}/bones/sv/image/benchmark/{self.dataset.name}_EXP.sh' ,shell=True)
            self.training_time += time.time()-time_start

            os.remove(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_explainer_BONES/checkpoints/last.ckpt')
            torch.cuda.empty_cache()
            if self.dataset.name == 'ImageNette':
                for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_explainer_BONES/checkpoints'):
                    if file.endswith(".ckpt"):
                        explainer_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/ImageNette_explainer_BONES/checkpoints/{file}'
                        break
            else:
                for file in os.listdir(f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_explainer_BONES/checkpoints'):
                    if file.endswith(".ckpt"):
                        surrogate_backbone = f'{os.getcwd()}/results/wandb_transformer_interpretability_project_BONES/Pet_explainer_BONES/checkpoints/{file}'
                        break

            explainer_VIT = Explainer(
                normalization="additive",
                normalization_class=_config["explainer_normalization_class"],
                activation="tanh",
                surrogate=surrogate_VIT,
                link='sigmoid' if _config["output_dim"]==1 else 'softmax',
                backbone_type="vit_tiny_patch16_224",
                download_weight=False,
                residual=[],
                load_path=explainer_backbone,
                target_type=_config["target_type"],
                output_dim=_config["output_dim"],

                explainer_head_num_attention_blocks=1,
                explainer_head_include_cls=True,
                explainer_head_num_mlp_layers=3,
                explainer_head_mlp_layer_ratio=4,
                explainer_norm=_config["explainer_norm"],

                efficiency_lambda=_config["explainer_efficiency_lambda"],
                efficiency_class_lambda=_config["explainer_efficiency_class_lambda"],
                freeze_backbone="all",

                checkpoint_metric=_config["checkpoint_metric"],
                optim_type=_config["optim_type"],
                learning_rate=_config["learning_rate"],
                weight_decay=_config["weight_decay"],
                decay_power=_config["decay_power"],
                warmup_steps=_config["warmup_steps"],
                load_path_state_dict=True).to(_config["gpus_explainer"][0])   
        self.explainer = explainer_VIT
    
    def compute(self, x, y):
        explanation=self.explainer(x.unsqueeze(0).to(self.explainer.device))
        explanation=explanation[0][0].T
        explanation_class=explanation[int(y)].detach().cpu().numpy()
        explanation_class=explanation_class.reshape(14, 14)
        svt=torch.tensor(explanation_class)
        return svt
    


class DeepExplainerModel():
    def __init__(self, dataset, config, classifierResPT, train_set, val_set, classifierPT, device, load, verbose):
        self.name='DeepExplainer'
        self.training_time=0
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_sv=[]
        background=[el[0] for el in train_set]
        background2 = torch.stack(background[:100])
        background2=background2.to(device)
        self.explainer = shap.DeepExplainer(classifierPT, background2)
        self.device = device
    
    def compute(self, x, y):
        # print(x.shape)
        shap_values_d = self.explainer.shap_values(x.unsqueeze(0).to(self.device)) 
        # print(shap_values_d.shape)
        shap_values_d = shap_values_d[0][0,:,:,:]
        shap_values_d = np.mean(shap_values_d, axis=0)
        # print(shap_values_d.shape)
        resized_shap_values = aggregate_and_resize_shap_values(shap_values_d)
        svd=torch.tensor(resized_shap_values)
        return svd



class GradientExplainerModel():
    def __init__(self, dataset, config, classifierResPT, train_set, val_set, classifierPT, device, load, verbose):
        self.name='GradientExplainer'
        self.training_time=0
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_sv=[]
        background=[el[0] for el in train_set]
        background2 = torch.stack(background[:100])
        background2=background2.to(device)
        self.explainer = shap.GradientExplainer(classifierPT, background2)
        self.device = device
    
    def compute(self, x, y):
        shap_values_g = self.explainer.shap_values(x.unsqueeze(0).to(self.device)) 
        shap_values_g = shap_values_g[0][0,:,:,:]
        shap_values_g = np.mean(shap_values_g, axis=0)
        resized_shap_values = aggregate_and_resize_shap_values(shap_values_g)
        svg=torch.tensor(resized_shap_values)
        return svg



class FastSHAPModel():
    def __init__(self, dataset, config, classifierResPT, train_set, val_set, classifierPT, device, load, verbose):
        self.name='FastSHAP'
        self.training_time=0
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_sv=[]
        self.dataset = dataset

        train_set_resize=CustomDataset([(torch.nn.functional.interpolate(el[0].unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False).squeeze(0), el[1]) for el in train_set])
        val_set_resize=CustomDataset([(torch.nn.functional.interpolate(el[0].unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False).squeeze(0), el[1]) for el in val_set])

        # Set up surrogate object
        print(f"{os.getcwd()}/models/{self.dataset.name}_surrogate.pt")
        if os.path.isfile(f"{os.getcwd()}/models/{self.dataset.name}_surrogate.pt") and load:
            if verbose:
                print("\t\tLOADING SURROGATE")
            surr = torch.load(f"models/{self.dataset.name}_surrogate.pt")
            surrogate=image_surrogate.ImageSurrogate(surr, dataset_name=self.dataset.name, width=56, height=56, superpixel_size=4)
        else:
            if verbose:
                print("\t\tTRAINING SURROGATE")

            surr = nn.Sequential(
                MaskLayer2d(value=0, append=True),
                resnet.ResNet18(in_channels=4, num_classes=self.dataset.num_classes),
            ).to(device)
            surrogate = image_surrogate.ImageSurrogate(surrogate=surr, dataset_name=self.dataset.name, width=56, height=56, superpixel_size=4)

            # Set up datasets
            train_surr = DatasetInputOnly(train_set_resize)
            val_surr = DatasetInputOnly(val_set_resize)
            original_model = nn.Sequential(classifierResPT, nn.Softmax(dim=1))
            # Train
            surrogate.train_original_model(
                train_surr,
                val_surr,
                original_model,
                batch_size=256,
                max_epochs=100,
                lr=1e-3,
                loss_fn=KLDivLoss(),
                lookback=10,
                bar=False,
                verbose=verbose
            )
            
            surr.cpu()
            torch.save(surr, f"models/{self.dataset.name}_surrogate.pt")
            surr.to(device)

            torch.cuda.empty_cache()

        self.surrogate = surrogate

        if verbose:
            print("\tTraining explainer model")

        if os.path.isfile(f"{os.getcwd()}/models/{self.dataset.name}_explainer.pt") and load:
            if verbose:
                print("\t\tLOADING EXPLAINER")
            explainer = torch.load(f"models/{self.dataset.name}_explainer.pt")
            fs=fastshap.FastSHAP(explainer, surrogate, link=nn.LogSoftmax(dim=1), dataset_name=self.dataset.name)
        else:
            if verbose:
                print("\t\tTRAINING EXPLAINER")
            explainer = unet.UNet(n_classes=10, num_down=3, num_up=1, num_convs=3).to(device)
            # Set up FastSHAP object
            fs = fastshap.FastSHAP(explainer=explainer, imputer=surrogate, link=nn.LogSoftmax(dim=1), dataset_name=self.dataset.name)#, normalization="additive")
            # Set up datasets
            fastshap_train = DatasetInputOnly(train_set_resize)
            fastshap_val = DatasetInputOnly(val_set_resize)

            # Train
            fs.train(
                fastshap_train,
                fastshap_val,
                batch_size=64,
                num_samples=2,
                max_epochs=200, ###############
                eff_lambda=1e-2,
                validation_samples=1,
                lr=1e-3,
                lookback=10,
                bar=False,
                verbose=verbose
            )

            torch.cuda.empty_cache()
            explainer.cpu()
            torch.save(explainer, f"models/{self.dataset.name}_explainer.pt")
            explainer.to(device)


        self.explainer = fs


    def compute(self, x, y):
        test_image_56=torch.nn.functional.interpolate(x.unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False)
        sv_fs=self.explainer.shap_values(test_image_56)
        svfs=sv_fs[0,int(y),:,:]
        svf=torch.tensor(svfs)
        return svf