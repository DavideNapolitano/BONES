import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from ..evaluation import Benchmark

class BarPlot():
    def __init__(self, benchmark, dsfn):
        self.benchmark = benchmark
        self.dsfn = dsfn

    def plot(self, index=None):
        explainers=self.benchmark.explainers_init[self.dsfn]
        self.dset = self.dsfn()
        X, Y, X_test, Y_test, feature_names, dataset=self.dset.get_data()
        if index is not None and index >= len(X):
            print('Sample index out of range')
            return

        plt.figure(figsize=(18, 9), dpi=300)
        width = 0.375
        colors_dict = {
            'Exact': 'tab:olive',
            'MonteCarlo': 'tab:green',
            'Unbiased KernelSHAP': 'tab:orange',
            'ShapleyRegression': 'tab:brown',
            'FastSHAP': 'tab:red',
            'DeepExplainer': 'tab:purple',
            'GradientExplainer': 'tab:blue',
            'DASP': 'tab:cyan'
        }
        shift_dict = {
            'Exact': -width * 0.875,
            'MonteCarlo': -width * 0.625,
            'Unbiased KernelSHAP': -width * 0.375,
            'ShapleyRegression': -width * 0.125,
            'FastSHAP': width * 0.125,
            'DeepExplainer': width * 0.375,
            'GradientExplainer': width * 0.625,
            'DASP': width * 0.875
        }
        if index == None:
            for k, expl in explainers.items():
                # plt.bar(np.arange(len(feature_names))+shift_dict[k], np.mean(expl.list_sv, axis=0), width=width/4, label=k, color=colors_dict[k])
                if k == 'ShapleyRegression':
                    plt.bar(np.arange(len(feature_names))+shift_dict[k], np.mean(expl.list_sv, axis=0), width=width/4, label="KernelSHAP", color=colors_dict[k])
                    plt.bar(np.arange(len(feature_names))+shift_dict['Unbiased KernelSHAP'], np.mean(expl.list_usv, axis=0), width=width/4, label='Unbiased KernelSHAP', color=colors_dict['Unbiased KernelSHAP'])
                else:
                    plt.bar(np.arange(len(feature_names))+shift_dict[k], np.mean(expl.list_sv, axis=0), width=width/4, label=k, color=colors_dict[k])
            
            plt.legend(fontsize=20)
            plt.tick_params(labelsize=18)
            plt.ylabel('SHAP Values', fontsize=18)
            # plt.title('BarPlot', fontsize=18)
            plt.xticks(np.arange(len(feature_names)), feature_names, rotation=35, rotation_mode='anchor', ha='right')

            plt.tight_layout()
            # save figure
            plt.savefig(f'barplot_{dataset}.pdf', dpi=300)
            plt.show()
        else:
            sample = X[index]
            sample= np.array([sample]) # add dimension to sample
            
            label = Y[index]
            IDX = index
            for k, expl in explainers.items():

                if k == 'ShapleyRegression':
                    out=expl.compute(IDX, sample, label, self.benchmark.kernelshap_iters)
                    # print(expl.name, out)   
                    plt.bar(np.arange(len(feature_names))+shift_dict[k], out[0], width=width/4, label="KernelSHAP", color=colors_dict[k])
                    plt.bar(np.arange(len(feature_names))+shift_dict['Unbiased KernelSHAP'], out[1], width=width/4, label='Unbiased KernelSHAP', color=colors_dict['Unbiased KernelSHAP'])
                else:
                    out=expl.compute(IDX, sample, label, self.benchmark.kernelshap_iters)[0]
                    # print(expl.name, out)   
                    plt.bar(np.arange(len(feature_names))+shift_dict[k], out, width=width/4, label=k, color=colors_dict[k])
            

            plt.legend(fontsize=20)
            plt.tick_params(labelsize=18)
            plt.ylabel('SHAP Values', fontsize=18)
            # plt.title('BarPlot', fontsize=18)
            plt.xticks(np.arange(len(feature_names)), feature_names, rotation=35, rotation_mode='anchor', ha='right')

            plt.tight_layout()
            # save figure
            plt.savefig(f'barplot_{dataset}.pdf', dpi=300)
            plt.show()

class TimeFeaturePlot():
    def __init__(self, benchmark):
        self.benchmark = benchmark

    def plot(self):
        dict={}
        dict_expl={}
        for dset_fn in self.benchmark.dataset:
            dset=dset_fn()
            X, Y, X_test, Y_test, feature_names, dataset=dset.get_data()
            explainers=self.benchmark.explainers_init[dset_fn]
            res={}
            for k, expl in explainers.items():
                res[expl.name] = np.mean(expl.list_time)
                if expl.name not in dict_expl:
                    dict_expl[expl.name] = []

                # if expl.name!=self.benchmark.ground_truth_name[dset_fn]:
                #     # if expl.name == 'ShapleyRegression':
                #     #     if len(expl.list_ul2)==0:
                #     #         res["Unbiased KernelSHAP"] = np.mean(expl.list_time)
                #     #         if "Unbiased KernelSHAP" not in dict_expl:
                #     #             dict_expl["Unbiased KernelSHAP"] = []
                #     #     else:
                #     #         res["Unbiased KernelSHAP"] = np.mean(expl.list_time)
                #     #         if "Unbiased KernelSHAP" not in dict_expl:
                #     #             dict_expl["Unbiased KernelSHAP"] = []
                #     #     res["KernelSHAP"] = np.mean(expl.list_time)
                #     #     if "KernelSHAP" not in dict_expl:
                #     #         dict_expl["KernelSHAP"]=[]
                #     # else:
                #     res[k] = [np.mean(expl.list_time)]
                # else:
                #     if expl.name == 'ShapleyRegression':
                #         res["KernelSHAP"] = [np.mean(expl.list_l2), np.mean(expl.list_time)]
                    
                #     else:
                #         res[expl.name] = [np.mean(expl.list_l2), np.mean(expl.list_time)]
            
            num_features=len(feature_names)
            dict[num_features] = res
        
        dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[0])}
        xticks = [k for k,v in dict.items()]
        for k,v in dict.items():
            for expl, time in v.items():
                dict_expl[expl].append(time)
        
        plt.figure(figsize=(16, 10), dpi=300)
        color_list=["olive", "green", "orange", "brown", "red", "purple", "blue", "cyan"]
        i=0
        for expl, times in dict_expl.items():
            # print(np.arange(1,len(times)), times)
            plt.plot(np.arange(1,len(times)+1), times, label=expl, color=color_list[i], linewidth=2.5)
            i+=1

        plt.xticks(np.arange(1,len(times)+1), xticks)

        # set y-axis to log scale
        plt.yscale('log')
        # Annotations
        plt.legend(fontsize=16, loc='upper left')
        plt.tick_params(labelsize=16)
        plt.xlabel('Number of Featuers', fontsize=18)
        plt.ylabel('Time (s)', fontsize=18)
        # plt.title('TimeFeature', fontsize=18)
        plt.tight_layout()
        # save the plot
        # plt.savefig(f'x_plot/timefeature_{dataset}.pdf', dpi=300)
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

        explainers=self.bench.explainers_init[self.dsfn]
        color_list=["olive", "green", "orange", "brown", "red", "purple", "blue", "cyan"]

        res={}
        steps=np.arange(0, self.number_samples+1, self.interval)
        steps[0]=1
        for k, expl in explainers.items():
            train_time=expl.time_train
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
        explainers=self.benchmark.explainers_init[self.dsfn]
        self.dset = self.dsfn()
        X, Y, X_test, Y_test, feature_names, dataset=self.dset.get_data()

        # plt.figure(figsize=(18, 9), dpi=300)

        # data_labels=['MonteCarlo', 'Unbiased KernelSHAP', 'KernelSHAP', 'FastSHAP', 'DeepExplainer', 'GradientExplainer', 'DASP']
        colors_list = ['green', 'orange', 'brown', 'red', 'purple', 'blue', 'cyan']

        res={}
        for k, expl in explainers.items():
            if expl.name!=self.benchmark.ground_truth_name[self.dsfn]:
                if expl.name == 'ShapleyRegression':
                    if len(expl.list_ul2)==0:
                        res["Unbiased KernelSHAP"] = [0, np.mean(expl.list_time)]
                    else:
                        res["Unbiased KernelSHAP"] = [np.mean(expl.list_ul2), np.mean(expl.list_time)]
                    res["KernelSHAP"] = [np.mean(expl.list_l2), np.mean(expl.list_time)]
                else:
                    res[k] = [np.mean(expl.list_l2), np.mean(expl.list_time)]
            else:
                if expl.name == 'ShapleyRegression':
                    res["KernelSHAP"] = [np.mean(expl.list_l2), np.mean(expl.list_time)]

        
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


        