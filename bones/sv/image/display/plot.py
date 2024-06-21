import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import shap

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import RandomSampler, BatchSampler

from ..explainers.fastshap.image_imputers import ImageImputer
from ..explainers.fastshap.utils import UniformSampler, DatasetRepeat, MaskLayer2d, KLDivLoss, DatasetInputOnly
from ..explainers.fastshap import resnet
from ..evaluation import Benchmark
# from image_imputers import ImageImputer
# from utils import UniformSampler, DatasetRepeat
from tqdm.auto import tqdm
from copy import deepcopy
import torch.nn as nn
import os

from .auc.evaluator import Evaluator

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




class AUCPlot():
    def __init__(self, benchmark, dsfn, num_samples):
        self.benchmark = benchmark
        self.dsfn = dsfn
        self.num_samples = num_samples

    def plot(self, verbose=False):
        device=self.benchmark.device
        train_set = self.benchmark.train_set
        val_set = self.benchmark.val_set
        train_set_resize=CustomDataset([(torch.nn.functional.interpolate(el[0].unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False).squeeze(0), el[1]) for el in train_set])
        val_set_resize=CustomDataset([(torch.nn.functional.interpolate(el[0].unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False).squeeze(0), el[1]) for el in val_set])


        if os.path.isfile(f'models/{self.dsfn().name}_evaluator.pt'):
            if verbose:
                print('Loading evaluator')
            evltr = torch.load(f'models/{self.dsfn().name}_evaluator.pt').to(device)
            evaluator = Evaluator(evltr, width=56, height=56, superpixel_size=4, dataset_name=self.dsfn().name)
        else:
            if verbose:
                print('Training evaluator')
            evltr = nn.Sequential(
                MaskLayer2d(value=0, append=True),
                resnet.ResNet18(in_channels=4, num_classes=self.dsfn().num_classes)
            ).to(device)

            # Set up surrogate object
            evaluator = Evaluator(evltr, width=56, height=56, superpixel_size=4, dataset_name=self.dsfn().name)

            # Set up datasets
            train_surr = DatasetInputOnly(train_set_resize)
            val_surr = DatasetInputOnly(val_set_resize)
            original_model = nn.Sequential(self.benchmark.classifierResPT, nn.Softmax(dim=1))
            # Train
            evaluator.train_original_model(
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

        labels=np.array([int(el[1]) for el in train_set][:self.num_samples])
        images=torch.stack([el[0] for el in train_set][:self.num_samples])
        preds=np.array([int(torch.argmax(torch.softmax(self.benchmark.classifierPT(img.unsqueeze(0).to(device)), dim=1))) for img in images])

        explainers=self.benchmark.explainers_init[self.dsfn().name]
        svs={}
        for k, expl in explainers.items():
            tmp=[]
            # print(expl.name)
            for img in (images):
                tmp.append(expl.compute(img, preds[0]).numpy())
            svs[expl.name]=np.array(tmp)

        down_images=[]
        for img in images:
            img_56=torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=False)
            down_images.append(img_56[0,:,:,:])
        down_images=torch.stack(down_images)


        names=list(svs.keys())
        results_exclude={}
        for name in names:
            results_exclude[name]=None

        for i,shapley_select in enumerate(list(svs.values())):
            shap_values_flat = shapley_select.reshape(shapley_select.shape[0], -1)
            masks_dict = {}
            up_mask_out={}
            for p in [99, 95, 90, 85, 75, 50, 25, 15, 10, 5, 1]:
                thresholds = np.percentile(shap_values_flat, p, axis=1)
                tmp=[sv < tr for sv, tr in zip(shapley_select, thresholds)]
                tmp_np=np.array(tmp).astype(int) 
                # upscale tmp
                # Assuming tmp is a torch.Tensor with shape (C, H, W)   
                tmp_t=torch.from_numpy(tmp_np)
                tmp_t=tmp_t.to(torch.float)
                tmp_up = torch.nn.functional.interpolate(tmp_t.unsqueeze(0), size=(56, 56), mode='bilinear', align_corners=False).squeeze(0)
                # print(tmp_up.shape)
                # print(tmp_np.shape)
                masks_dict[str(100-p)] = tmp_np
                up_mask_out[str(100-p)] = tmp_up

            res=[]
            for p, masks in up_mask_out.items():
                if len(masks.shape) < 4:
                    masks = np.expand_dims(masks, 1)
                # downscale images
                masked_images = down_images * masks
                S=torch.ones(masked_images.shape[0], evaluator.num_players)
                evals=evaluator(masked_images.to(device),S.to(device))
                evals=evals.detach().cpu().numpy()
                argmax = torch.argmax(torch.from_numpy(evals), dim=1)
                local_preds=torch.from_numpy(preds)
                matches = (argmax == local_preds).float()

                # Calculate accuracy
                accuracy = matches.sum() / len(preds)
                res.append(accuracy.item())
            
            results_exclude[names[i]]=res
        
        plt.figure(figsize=(7, 7), dpi=200)
        colors=['green', 'orange', 'brown', 'red', 'purple', 'blue', 'cyan']
        i=0
        for k,v in results_exclude.items():
            plt.plot([int(el) for el in up_mask_out.keys()], v, label=k, color=colors[i], linewidth=3)
            i+=1
        # plt.plot([int(el) for el in up_mask_out.keys()], res)

        plt.legend(fontsize=16)
        plt.tick_params(labelsize=16)
        plt.xticks(np.arange(0, 101, 20))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('Exclusion %', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.title('Exclusion Curve', fontsize=20)
        plt.tight_layout()
        # plt.savefig('exclusion_curve.pdf')
        plt.show()


        results_include={}
        for name in names:
            results_include[name]=None

        for i,shapley_select in enumerate(list(svs.values())):
            shap_values_flat = shapley_select.reshape(shapley_select.shape[0], -1)
            masks_dict = {}
            up_mask_in={}
            for p in [99, 95, 90, 85, 75, 50, 25, 15, 10, 5, 1]:
                thresholds = np.percentile(shap_values_flat, p, axis=1)
                tmp=[sv >= tr for sv, tr in zip(shapley_select, thresholds)]
                tmp_np=np.array(tmp).astype(int) 
                # upscale tmp
                # Assuming tmp is a torch.Tensor with shape (C, H, W)   
                tmp_t=torch.from_numpy(tmp_np)
                tmp_t=tmp_t.to(torch.float)
                tmp_up = torch.nn.functional.interpolate(tmp_t.unsqueeze(0), size=(56, 56), mode='bilinear', align_corners=False).squeeze(0)
                # print(tmp_up.shape)
                # print(tmp_np.shape)
                masks_dict[str(100-p)] = tmp_np
                up_mask_in[str(100-p)] = tmp_up

            res=[]
            for p, masks in up_mask_in.items():
                if len(masks.shape) < 4:
                    masks = np.expand_dims(masks, 1)
                # downscale images
                masked_images = down_images * masks
                S=torch.ones(masked_images.shape[0], evaluator.num_players)
                # print(masked_images.shape, S.shape)
                evals=evaluator(masked_images.to(device),S.to(device))
                evals=evals.detach().cpu().numpy()

                argmax = torch.argmax(torch.from_numpy(evals), dim=1)

                local_preds=torch.from_numpy(preds)
                matches = (argmax == local_preds).float()

                # Calculate accuracy
                accuracy = matches.sum() / len(preds)
                res.append(accuracy.item())

            results_include[names[i]]=res

        # plot res. on the x-axis the percentage of the mask, on the y-axis the accuracy. xticks every 20, yticks every 0.1
        plt.figure(figsize=(7, 7), dpi=200)

        i=0
        for k,v in results_include.items():
            plt.plot([int(el) for el in up_mask_in.keys()], v, label=k, color=colors[i], linewidth=3)
            i+=1
        # plt.plot([int(el) for el in up_mask_out.keys()], res)

        plt.legend(fontsize=16)
        plt.tick_params(labelsize=16)
        plt.xticks(np.arange(0, 101, 20))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('Inclusion %', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.title('Inclusion Curve', fontsize=20)
        plt.tight_layout()
        # plt.savefig('inclusion_curve.pdf')
        plt.show()


class ImagePlot():
    def __init__(self, benchmark, dsfn):
        self.benchmark = benchmark
        self.dsfn = dsfn
    
    def plot(self, index):
        data=self.benchmark.train_set
        if index > len(data):
            print("Index out of range")
            return
        data=data[index]
        sample=data[0]
        label=data[1]
        explainers=self.benchmark.explainers_init[self.dsfn().name]
        svs={}
        for k, expl in explainers.items():
            svs[expl.name]=expl.compute(sample, label)
        
        values=list(svs.values())

        fig, axes = plt.subplots(1, len(values) + 1, figsize=(20, 9))
        axes[0].imshow(sample.permute(1, 2, 0), vmin=0, vmax=1)#, vmin=0, vmax=1)#, cmap=plt.get_cmap('gray'))
        axes[0].set_yticklabels([])
        axes[0].set_xticklabels([])
        axes[0].set_frame_on(False)
        axes[0].tick_params(length=0)
        # get the image in greyscale

        im=sample.permute(1, 2, 0)
        im_gray = (0.2989 * im[:,:,0] + 0.5870 * im[:,:,1] + 0.1140 * im[:,:,2]) # rgb to gray

        names=list(svs.keys())
        for col in range(len(values)):
            ex=values[col].numpy()
            abs_vals = np.abs(ex).flatten()
            max_val = np.max(abs_vals)
            if max_val == 0:
                max_val += 1e-8
            axes[col+1].set_title(names[col], fontsize=20)
            # m=values[col].max()
            # axarr[col + 1].imshow(greyscale_image, cmap=plt.get_cmap('gray'), alpha=0.3)
            # axarr[col + 1].imshow(values[col], cmap=shap.plots.colors.red_transparent_blue, vmin=-m, vmax=m)
            axes[col+1].imshow(im_gray, cmap=plt.get_cmap('gray'), alpha=0.3, extent=(-1, ex.shape[1], ex.shape[0], -1))
            im = axes[col+1].imshow(ex, cmap=shap.plots.colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
            axes[col+1].axis('off')

        # save figure with tight layout
        plt.tight_layout()
        # plt.savefig('comparison.pdf')
        plt.show()


class TimeSamplePlot():
    def __init__(self, benchmark, dataset, number_samples, interval, sample_method):
        self.benchmark = benchmark
        self.dsfn = dataset
        self.number_samples = number_samples
        self.interval = interval
        self.sample_method = sample_method
        self.bench=None
        self.run()

    def run(self):
        self.bench=Benchmark(explainers=self.benchmark.explainers, dataset=[self.dsfn], ground_truth=self.benchmark.ground_truth, \
            metrics=self.benchmark.metrics, num_samples=self.number_samples, sample_method=self.sample_method)
        self.bench.run(verbose=True, load=True)
        
    
    def plot(self):
        # create a new benchmark object
        # bench=Benchmark(explainers=self.benchmark.explainers, dataset=[self.dsfn], ground_truth=self.benchmark.ground_truth, \
        #     metrics=self.benchmark.metrics, num_samples=self.number_samples, sample_method=self.sample_method).run(verbose=True, load=True)

        explainers=self.bench.explainers_init[self.dsfn().name]
        color_list=["olive", "green", "orange", "brown", "red", "purple", "blue", "cyan"]

        res={}
        steps=np.arange(0, self.number_samples+1, self.interval)
        steps[0]=1
        for k, expl in explainers.items():
            train_time=expl.training_time
            inf_time=[]
            for i in steps:
                inf_time.append(np.sum(expl.list_time[:i])+ train_time) 
            res[expl.name] = inf_time

        plt.figure(figsize=(16, 10), dpi=300)
        i=0
        for k,v in res.items():
            plt.plot(steps, v, label=k, color=color_list[i], linewidth=2.5)
            i+=1
        
        # set y-axis to log scale
        plt.yscale('log')
        # Annotations
        plt.legend(fontsize=16)
        plt.tick_params(labelsize=16)
        plt.xlabel('Number of samples', fontsize=18)
        plt.ylabel('Time (s)', fontsize=18)
        plt.tight_layout()
        plt.show()


class QuadrantPlot():
    def __init__(self, benchmark, dsfn):
        self.benchmark = benchmark
        self.dsfn = dsfn

    def plot(self):
        explainers=self.benchmark.explainers_init[self.dsfn().name]
        self.dset = self.dsfn()
        # X, Y, X_test, Y_test, feature_names, dataset=self.dset.get_data()

        # plt.figure(figsize=(18, 9), dpi=300)

        # data_labels=['MonteCarlo', 'Unbiased KernelSHAP', 'KernelSHAP', 'FastSHAP', 'DeepExplainer', 'GradientExplainer', 'DASP']
        colors_list = ['green', 'orange', 'brown', 'red', 'purple', 'blue', 'cyan']

        res={}
        for k, expl in explainers.items():
            if expl.name!=self.benchmark.ground_truth_name[self.dsfn().name]:
                res[k] = [np.mean(expl.list_l2), np.mean(expl.list_time)]

        
        list_times = [res[k][1] for k in res]
        list_val = [res[k][0] for k in res]

        list_times = np.array(list_times)
        X_std = (list_times - list_times.min(axis=0)) / (list_times.max(axis=0) - list_times.min(axis=0))
        X_scaled = X_std * (1 - 0) + 0
        list_times = X_scaled

        list_val = np.array(list_val)
        X_std = (list_val - list_val.min(axis=0)) / (list_val.max(axis=0) - list_val.min(axis=0))
        X_scaled = X_std * (1 - 0) + 0
        list_val = X_scaled
        list_val=1-list_val

        x=list_times
        y=list_val
        data_labels = [k for k in res]
        colors=[colors_list[i] for i in range(len(data_labels))]

        data = {
            'Name': data_labels,
            'Time': x,  # X-axis values
            'Performance': y,  # Y-axis values
            'size': np.ones(len(data_labels))*10,
        }

        df = pd.DataFrame(data)

        # Create the scatter plot
        fig = px.scatter(
            df, 
            x='Time', 
            y='Performance', 
            text='Name',

            hover_name='Name',
            hover_data={'Time': True, 'Performance': True, 'Name': False, 'size': False},
            color='Name',
            size='size',
        )

        fig.update_traces(textposition='top center')
        # fig.upda

        # Update layout to position quadrants and add lines
        fig.update_layout(
            shapes=[
                dict(
                    type='line',
                    x0=0.5, x1=0.5,
                    y0=0, y1=1,
                    line=dict(color='Black', width=1)
                ),
                dict(
                    type='line',
                    x0=0, x1=1,
                    y0=0.5, y1=0.5,
                    line=dict(color='Black', width=1)
                ),
                dict(
                    type='line',
                    x0=0, x1=0,
                    y0=0, y1=1,
                    line=dict(color='Black', width=1)
                ),
                dict(
                    type='line',
                    x0=1, x1=1,
                    y0=0, y1=1,
                    line=dict(color='Black', width=1)
                ),
                dict(
                    type='line',
                    x0=0, x1=1,
                    y0=0, y1=0,
                    line=dict(color='Black', width=1)
                ),
                dict(
                    type='line',
                    x0=0, x1=1,
                    y0=1, y1=1,
                    line=dict(color='Black', width=1)
                )
            ],
            xaxis=dict(range=[-0.01, 1.05],
                title=dict(
                    text='Time',  # Set X-axis title text
                    # standoff=400,  # Add space between the axis and the title
                ),
                showticklabels=False,  # Hide Y-axis ticks and labels
                showgrid=False,        # Hide Y-axis grid lines
                zeroline=False,        # Hide Y-axis zero line
                showline=False,
                automargin=True,          
                ),  # Set X-axis range
            yaxis=dict(range=[-0.1,1.05], 
                showticklabels=False,  # Hide Y-axis ticks and labels
                showgrid=False,        # Hide Y-axis grid lines
                zeroline=False,        # Hide Y-axis zero line
                showline=False         
            ),      # Set Y-axis range
            width=1000,  # Set the width of the plot
            height=1000,  # Set the height of the plot to be equal to the width
            autosize=True,  # Ensure the plot size does not auto adjust
            margin=dict(l=40, r=40, b=40, t=40),
            plot_bgcolor='white',  # Set the plot background to white
            paper_bgcolor='white',  # Set the figure background to white
            showlegend=False,
            # autoscalse=True
        )

        # Add annotations for quadrant labels
        fig.add_annotation(x=0.25, y=1.025, text='Low', showarrow=False)
        fig.add_annotation(x=-0.04, y=0.25, text='Low', showarrow=False)
        fig.add_annotation(x=-0.04, y=0.75, text='High', showarrow=False)
        fig.add_annotation(x=0.75, y=1.025, text='High', showarrow=False)

        fig.show()

        