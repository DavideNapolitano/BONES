import shap
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, Conv2D, MaxPooling2D
from keras import regularizers
from keras import backend as K
from tqdm import tqdm
import time
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# from ..explainers import MonteCarlo, ShapleyRegression, FastSHAP, DASP, DeepExplainer, Exact, MarginalExtension
from ..explainers.models import DASPModel, MonteCarloModel, DeepExplainerModel, GradientExplainerModel, ExactExplainerModel, ShapleyRegressionModel, FastSHAPModel
from .tabulate import tabulate

def is_tensorflow_model(model):
    return isinstance(model, keras.Model)

from sklearn.base import BaseEstimator

def is_sklearn_classifier(model):
    return isinstance(model, BaseEstimator)

import torch.nn as nn

def is_pytorch_model(model):
    return isinstance(model, nn.Module)

class Benchmark():
    def __init__(self, dataset, explainers, metrics, ground_truth, num_samples, classifier=None, sample_method='random'):
        self.dataset = dataset
        self.explainers = explainers
        self.explainers_init = {}
        self.metrics = metrics
        self.ground_truth = ground_truth
        self.ground_truth_name = {}
        self.num_samples = num_samples
        self.sample_method = sample_method
        self.kernelshap_iters = 128 # default value
        self.classifier = classifier

        check_classifier = is_tensorflow_model(self.classifier)
        # if classifier is not tensorflow model and DASPModel is in explainers, print error and exit
        if check_classifier==False and DASPModel in explainers:
            print("Error: DASPModel requires a tensorflow model as classifier")
            exit(1)
        

    def run(self, verbose=False, load=False): # CURRENTLY THE GROUND TRUTH IS THE SAME FOR EACH DATASET
        for dset_fn in self.dataset:
            dset = dset_fn()
            
            X, Y, X_test, Y_test, feature_names, dataset=dset.get_data()
            if verbose:
                print("-"*100)
                print("Running dataset:", dataset)

            if X_test is None:
                num_features = X.shape[1]
                X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
                if verbose:
                    print(X.shape, Y.shape, feature_names)
                # data=X_train
                ss = StandardScaler()
                ss.fit(X_train)
                X_train_ss = ss.transform(X_train)
                X_val_ss = ss.transform(X_val)
            else:
                num_features = X.shape[1]
                X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
                if verbose:
                    print(X.shape, Y.shape, X_test.shape, Y_test.shape, feature_names)
                # data=X_train
                ss = StandardScaler()
                ss.fit(X_train)
                X_train_ss = ss.transform(X_train)
                X_val_ss = ss.transform(X_val)
                X_test_ss = ss.transform(X_test)


            if verbose:
                print("\nRunning Black-box Model")

            if self.classifier == None: # if classifier is not provided, create a simple neural network
                bbmodel = Sequential()
                bbmodel.add(Dense(64, activation='relu', input_shape=(num_features,)))
                bbmodel.add(Dropout(0.5))
                bbmodel.add(Dense(64, activation='relu'))
                bbmodel.add(Dropout(0.5))
                # model.add(Dense(1, activation='sigmoid'))
                bbmodel.add(Dense(2))
                bbmodel.add(Activation('softmax'))

                # Compile the model
                bbmodel.compile(optimizer='adam',
                            #   loss='binary_crossentropy',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

                # Train the model
                bbmodel.fit(X_train_ss, Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val), verbose=0)

                # Evaluate the model
                loss, accuracy = bbmodel.evaluate(X_val_ss, Y_val, verbose=0)
                if verbose:
                    print('\tAccuracy:', accuracy)
            else: # if classifier is provided, train it
                # if self.classifier is a tensorflow model
                if is_tensorflow_model(self.classifier):
                    bbmodel = self.classifier
                    bbmodel.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
                    
                    # Train the model
                    bbmodel.fit(X_train_ss, Y_train, epochs=50, batch_size=32, validation_data=(X_val_ss, Y_val), verbose=0)

                    # Evaluate the model
                    loss, accuracy = bbmodel.evaluate(X_val_ss, Y_val, verbose=0)
                    if verbose:
                        print('\tAccuracy:', accuracy)

                # if self.classifier is a sklearn model
                elif is_sklearn_classifier(self.classifier):
                    bbmodel = self.classifier
                    bbmodel.fit(X_train_ss, Y_train)
                    accuracy = bbmodel.predict(X_val_ss, Y_val)
                    if verbose:
                        print('\tAccuracy:', accuracy)
                
                # if self.classifier is a pytorch model, train it
                elif is_pytorch_model(self.classifier):
                    bbmodel = self.classifier
                    bbmodel.train()
                    criterion = nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(bbmodel.parameters(), lr=0.001)
                    for epoch in range(50):
                        optimizer.zero_grad()
                        output = bbmodel(torch.tensor(X_train_ss, dtype=torch.float32))
                        loss = criterion(output, torch.tensor(Y_train, dtype=torch.long))
                        loss.backward()
                        optimizer.step()
                    bbmodel.eval()
                    output = bbmodel(torch.tensor(X_val_ss, dtype=torch.float32))
                    _, predicted = torch.max(output, 1)
                    accuracy = accuracy_score(Y_val, predicted)
                    if verbose:
                        print('\tAccuracy:', accuracy)
                else:
                    print("Error: Classifier is not supported")
                    exit(1)


            kernelshap_iters = self.kernelshap_iters
            df = pd.DataFrame(X_train_ss, columns=feature_names)

            
            ground_model = self.ground_truth(df, bbmodel, 1000, num_features, X_train_ss, X_val_ss, load, dataset)
            print("\nGround Truth:", ground_model.name)
            self.ground_truth_name[dset_fn]=ground_model.name

            # initialize explainers as a dictionary name:explainer
            explainers = {}
            explainers[ground_model.name] = ground_model
            print("\nInitializing Explainers")
            for explainer in self.explainers:
                print("\tExplainer:", explainer) #still not initialized
                exp = explainer(df, bbmodel, 1000, num_features, X_train_ss, X_val_ss, load, dataset)
                explainers[exp.name]=exp
            

            print("\nExplainers:", explainers.keys())

            self.explainers_init[dset_fn] = explainers

            # initialize metrics as a dictionary name:metric
            metrics = {}
            for metric in self.metrics:
                mtr=metric()
                metrics[mtr.name]=mtr

            print("Metrics:", metrics.keys())

            if self.num_samples>len(X_train_ss):
                if self.sample_method == "random":
                    # sample random samples from the dataset
                    indexes=[]
                    for i in range(self.num_samples):
                        indexes.append(random.sample(list(np.arange(len(X_train_ss))), 1)[0])
                        
                    DATA=[]
                    LABELS=[]
                    for idx in indexes:
                        DATA.append(X_train_ss[idx])
                        LABELS.append(Y_train[idx])
                    DATA=np.array(DATA)
                    LABELS=np.array(LABELS)
                else:
                    print("Error: Number of samples is greater than the number of samples in the dataset")
                    print("Please set sample_method to 'random' or reduce the number of samples")
                    exit(1)
            else:
                DATA=X_train_ss
                LABELS=Y_train
            
            print("\nRunning Explainers")
            for IDX in tqdm(range(len(DATA[:self.num_samples]))):
                sample=DATA[IDX:IDX+1]
                label=LABELS[IDX:IDX+1][0]
                # print(sample, label)

                for k, expl in explainers.items():
                    time_start = time.time()
                    out=expl.compute(IDX, sample, label, kernelshap_iters)
                    time_end = time.time()
                    expl.list_time.append(time_end-time_start)
                    if len(out)>1:
                        expl.list_usv.append(out[0])
                        expl.list_sv.append(out[1])
                    else:
                        expl.list_sv.append(out[0])
                
                if ground_model.name == "ShapleyRegression":
                    ground_sv=ground_model.list_usv[IDX]
                    for k, mtr in metrics.items():
                        for k, expl in explainers.items():
                            sv_comparison=expl.list_sv[IDX]
                            res=mtr.compute(sv_comparison, ground_sv)
                            if mtr.name == 'L1':
                                expl.list_l1.append(res)
                            if mtr.name == 'L2':
                                expl.list_l2.append(res)
                            if mtr.name == 'Kendall':
                                expl.list_kendall.append(res)
                else:
                    ground_sv=ground_model.list_sv[IDX]
                    for k, mtr in metrics.items():
                        for k, expl in explainers.items():
                            if expl.name != ground_model.name:
                                if expl.name == 'ShapleyRegression':
                                    sv_comparison=expl.list_usv[IDX]
                                    res=mtr.compute(sv_comparison, ground_sv)
                                    if mtr.name == 'L1':
                                        expl.list_ul1.append(res)
                                    if mtr.name == 'L2':
                                        expl.list_ul2.append(res)
                                    if mtr.name == 'Kendall':
                                        expl.list_ukendall.append(res)
                                sv_comparison=expl.list_sv[IDX]
                                res=mtr.compute(sv_comparison, ground_sv)
                                if mtr.name == 'L1':
                                    expl.list_l1.append(res)
                                if mtr.name == 'L2':
                                    expl.list_l2.append(res)
                                if mtr.name == 'Kendall':
                                    expl.list_kendall.append(res)
            
            if verbose:
                print("-"*100)
            

    def print_results(self, dsfn):
        explainers=self.explainers_init[dsfn]
        table = [['Method', 'Time Training', 'Time Inference', 'Time Total', 'L1', 'L2', 'Rank']]
        for k, expl in explainers.items():
            if expl.name == 'ShapleyRegression':
                if len(expl.list_ul1)==0:
                    table.append(["Unbiased KernelSHAP", expl.time_train, np.mean(expl.list_time), np.mean(expl.list_time)+expl.time_train, 0, 0, 0])
                else:
                    table.append(["Unbiased KernelSHAP", expl.time_train, np.mean(expl.list_time), np.mean(expl.list_time)+expl.time_train, np.mean(expl.list_ul1), np.mean(expl.list_ul2), np.mean(expl.list_ukendall)])
                table.append(["KernelSHAP", expl.time_train, np.mean(expl.list_time), np.mean(expl.list_time)+expl.time_train, np.mean(expl.list_l1), np.mean(expl.list_l2), np.mean(expl.list_kendall)])
            else:
                if len(expl.list_l1)==0:
                    table.append([expl.name, expl.time_train, np.mean(expl.list_time), np.mean(expl.list_time)+expl.time_train, 0, 0, 0])
                else:
                    table.append([expl.name, expl.time_train, np.mean(expl.list_time), np.mean(expl.list_time)+expl.time_train, np.mean(expl.list_l1), np.mean(expl.list_l2), np.mean(expl.list_kendall)])


        print(tabulate(table, headers='firstrow', headersglobalalign='center', tablefmt='fancy_grid', colalign=('center','center','global','global','global','global','global')))