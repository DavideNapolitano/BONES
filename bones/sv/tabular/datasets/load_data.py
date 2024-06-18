import pandas as pd
import sys
import os
print(os.getcwd())
sys.path.append('/')
sys.path.append('/shapleyeval/models')
sys.path.append('/shapleyeval/save')
sys.path.append("/shapleyeval/data")
from shapleyeval.tabular.datasets import Monks, Census, Magic, Credit, Wbcd, Diabetes, Heart, Mozilla, Phoneme, Bank

def load_data(dataset):
    if dataset == 'Census':
        return Census().get_data()
    if dataset == 'Monks':
        return Monks().get_data()
    if dataset == 'Magic':
        return Magic().get_data()
    if dataset == 'Credit':
        return Credit().get_data()
    if dataset == 'Wbcd':
        return Wbcd().get_data()
    if dataset == 'Diabetes':
        return Diabetes().get_data()
    if dataset == 'Heart':
        return Heart().get_data()
    if dataset == 'Mozilla':
        return Mozilla().get_data()
    if dataset == 'Phoneme':
        return Phoneme().get_data()
    if dataset == 'Bank':
        return Bank().get_data()