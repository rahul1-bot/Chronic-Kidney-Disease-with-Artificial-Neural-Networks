from __future__ import annotations

__authors__: list[str] = ['Rahul_Sawhney', 'Prateeksha_Agrawal']
#$ exec(False) if not __pipeline_1__.__dict__() or any('Rahul_Sawhney', 'Prateeksha_Agrawal',) in __authors__

__doc__ = r'''
    Project Topic -> Chronic Kidney Disease with Multi-Layer Perceptron Model 

    Project Abstract -> ...

    project Flow -> 1) pipeline_1.py: Data Engineering
                    2) pipeline_2.py: Machine Learning
                    3) main.ipynb
'''

#@: Pipeline_1: Data_Engineering
    #@: class KidneyDataset
    #       : __init__                          -> None
    #       : __repr__                          -> str(dict[str, str])
    #       : __str__                           -> str(dict[str, str])
    #       : __len__                           -> tuple[int, ...]
    #       : __getitem__                       -> pd.Series
    #
    #@: class KidneyAnalysis
    #       : __repr__                          -> str(dict[str, str])
    #       : __str__                           -> str(dict[str, str])
    #       : data_characteristics()            -> Generator[_text, None, None]
    #       : data_unique_values()              -> Generator[_text, None, None]
    #       : target_classification_count()     -> pd.Series  
    #       : replace_incorrect_values()        -> pd.DataFrame
    #       : target_distribution()             -> _plot
    #       : histograms_numeric_features()     -> _plot
    #
    #@: class KidneyPreprocess
    #       : __repr__                          -> str(dict[str, str])
    #       : __str__                           -> str(dict[str, str])
    #       : replace_values()                  -> pd.DataFrame
    #       : change_object_to_str()            -> pd.DataFrame
    #       : encode_features()                 -> pd.DataFrame
    #       : impute_nan_values()               -> tuple[pd.DataFrame, pd.Series]
    #       : missing_values_percentage()       -> Generator[pd.Series, None, None]
    #
    #@: if __name__.__contains__('__main__')


# python imports
from typing import Any, NewType, Container, Optional, Generator
import warnings, os
warnings.filterwarnings(action= 'ignore')

# Data Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Scripting ML
from sklearn.impute import KNNImputer

# DL Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# torch typing scripts
_path   = NewType('_path', Any)
_loader = NewType('_loader', Any)
_model  = NewType('_model', Any)
_loss   = NewType('_loss', Any)
_optimizer = NewType('_optimizer', Any)
_transform = NewType('_transform', Any)
_text = NewType('_text', Any)
_imputer = NewType('_Imputer', Any)


#@: Class KidneyDataset
class KidneyDataset(torch.utils.data.Dataset):
    def __init__(self, path: _path, target: Optional[str] = None, transform: Optional[_transform] = None) -> None:
        self.dataset: pd.DataFrame = pd.read_csv(path, sep= ',')
        if target:
            self.X: pd.DataFrame = self.dataset.drop(target, axis= 1)
            self.y: pd.DataFrame = self.dataset[target]
        if transform:
            self.X: np.ndarray = transform(self.X)


    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        }) 


    __str__ = __repr__


    def __len__(self) -> int:
        return self.dataset.shape[0]
    

    def __getitem__(self, index: int) -> pd.Series:
        return self.dataset.iloc[index]




#@: Class DataAnalysis
class KidneyAnalysis:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    __str__ = __repr__


    @classmethod
    def rename_columns(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        col_dict: dict[str, str] = {
            'bp' : 'blood_pressure',  'sg'  : 'specific_gravity', 'al' : 'albumin', 
            'su' : 'sugar',           'rbc' : 'red_blood_cells',  'pc' : 'pus_cell', 
            'pcc': 'pus_cell_clumps', 'ba'  : 'bacteria',         'bgr': 'blood_glucose_random', 
            'bu' : 'blood_urea',      'sc'  : 'serum_creatinine', 'sod': 'sodium', 
            'pot': 'potassium',       'hemo': 'hemoglobin',       'pcv': 'packed_cell_volume',
            'wc' : 'white_blood_cell_count',  
            'rc' : 'red_blood_cell_count',
            'htn': 'hypertension', 
            'dm' : 'diabetes_mellitus',
            'cad': 'coronary_artery_disease', 
            'appet':'appetite', 'pe':'pedal_edema', 'ane':'anemia'
        }

        dataset.rename(columns= col_dict, inplace=True)
        return dataset



    @classmethod
    def data_characteristics(cls, dataset: pd.DataFrame, 
                                  info: Optional[bool] = False,
                                  summary: Optional[bool] = False) -> Generator[_text, None, None]:
        yield f'Dataset Shape: {dataset.shape}'
        yield f'Dataset features: {dataset.columns.values}'
        if info:
            yield f'Dataset Info: {dataset.info()}'
        yield f'Dataset types: {dataset.dtypes}'
        if summary:
            yield dataset.describe().T
        yield f'Null Values in Dataset: {dataset.isnull().sum()}'



    @classmethod
    def data_unique_values(cls, dataset: pd.DataFrame) -> Generator[_text, None, None]:
        for column in dataset.columns:
            yield f'Unique Value in {column} :\n{dataset[column].unique()}'


    @classmethod
    def target_classification_count(cls, dataset: pd.DataFrame, target: str) -> pd.Series:
        yield f'Total count of the Prediction column :\n{dataset[target].value_counts()}'
    


    @classmethod
    def target_distribution(cls, dataset: pd.DataFrame, target: str) -> _plot:
        sns.countplot(x= target, data= dataset)
        plt.xlabel(target)
        plt.ylabel('Count')
        plt.title('Target Classification Distribution')
        plt.show()


    @classmethod
    def histograms_numeric_features(cls, dataset: pd.DataFrame, numeric_features: list[str]) -> _plot:
        dataset.hist(numeric_features, figsize = (15, 8))
        plt.show()



#@: class Data Preprocess
class KidneyPreprocess:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    
    __str__ = __repr__


    @classmethod
    def replace_values(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['diabetes_mellitus'] = dataset['diabetes_mellitus'].replace(to_replace= {
            '\tno': 'no', '\tyes': 'yes', ' yes': 'yes'
        })
        dataset["white_blood_cell_count"] = dataset["white_blood_cell_count"].replace(to_replace= {
            '\t?': np.nan, '\t8400': '8400'
        })
        dataset["red_blood_cell_count"]   = dataset["red_blood_cell_count"].replace(to_replace= {
            '\t?': np.nan
        })
        dataset['coronary_artery_disease'] = dataset['coronary_artery_disease'].replace(to_replace= {
            '\tno': 'no'
        })
        dataset['classification'] = dataset['classification'].replace(to_replace= {
            'ckd\t': 'ckd'
        })
        dataset["packed_cell_volume"] = dataset["packed_cell_volume"].replace(to_replace= {
            '\t?': np.nan
        })
        return dataset


    @classmethod
    def change_object_to_str(cls, dataset: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            dataset[col] = dataset[col].astype(str)
        return dataset



    @classmethod
    def encode_features(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset['red_blood_cells'] = dataset['red_blood_cells'].replace(to_replace= {
            'normal': 1, 'abnormal': 0
        })
        dataset['pus_cell'] = dataset['pus_cell'].replace(to_replace= {
            'normal': 1, 'abnormal': 0
        })
        dataset['pus_cell_clumps'] = dataset['pus_cell_clumps'].replace(to_replace= {
            'notpresent': 0, 'present': 1
        })
        dataset['bacteria'] = dataset['bacteria'].replace(to_replace= {
            'notpresent': 0, 'present': 1
        })
        dataset['hypertension'] = dataset['hypertension'].replace(to_replace= {
            'no': 0, 'yes': 1
        })
        dataset['diabetes_mellitus'] = dataset['diabetes_mellitus'].replace(to_replace= {
            'no': 0, 'yes': 1
        })
        dataset['coronary_artery_disease'] = dataset['coronary_artery_disease'].replace(to_replace= {
            'no': 0, 'yes': 1
        })
        dataset['pedal_edema'] = dataset['pedal_edema'].replace(to_replace= {
            'no': 0, 'yes': 1
        })
        dataset['anemia'] = dataset['anemia'].replace(to_replace= {
            'no': 0, 'yes': 1
        })
        dataset['appetite'] = dataset['appetite'].replace(to_replace= {
            'poor': 0, 'good': 1
        })
        dataset['classification'] = dataset['classification'].replace(to_replace = {
            'ckd' : 1, 'notckd': 0
        }) 
        return dataset



    @classmethod
    def impute_nan_values(cls, dataset: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
        y: pd.Series = dataset[target]
        imputer: _imputer = KNNImputer(n_neighbors= 5, weights= 'uniform', metric= 'nan_euclidean')
        impute_cols: list[str] = list(
            set(dataset.columns) - set(['classification'])
        )
        imputer.fit(dataset[impute_cols])
        X: pd.DataFrame = pd.DataFrame(
            data= imputer.transform(dataset[impute_cols]),
            columns= impute_cols
        )
        return X, y



    @classmethod
    def missing_values_percentage(cls, X: pd.DataFrame, y: pd.Series) -> Generator[pd.Series, None, None]:
        yield round(
            (X.isnull().sum() * 100/ len(X)), 2
        ).sort_values(ascending=False)
        yield round(
            (y.isnull().sum() * 100/ len(y)), 2
        ).sort_values(ascending=False)
        




# Driver code
if __name__.__contains__('__main__'):
    # Module Usage: from pipeline_1 import *
    pass