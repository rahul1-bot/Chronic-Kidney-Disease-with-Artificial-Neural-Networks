# Pipeline 1: Data Engineering
from __future__ import annotations
__author__: str = 'Rahul_Sawhney'
#$ exec(False) if not __pipeline1__.__dict__ or 'Rahul_Sawhney' != __author__

__doc__ = r"""
    Project Abstract: ...

    Project Control Flow: ...

    PipeLine1 Flow: 1) Class DataSet
                    2) Class DataAnalysis
                    3) Class DataPrerpocess


    """

# python Imports
from typing import Any, ClassVar, NewType, Optional, Generator
import os, warnings
warnings.filterwarnings('ignore')

# Data Anaalysis Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
style.use('fivethirtyeight')

# DL 
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F

# Torch-Typing Scripts
_model = NewType('_model', Any)
_path = NewType('_path', Any)
_transform = NewType('_transform', Any)
_plot = NewType('_plot', Any)


#@:  ---- Class DataSet ----
class KidneyDataset(Dataset):
    def __init__(self, data_path: _path, target: Optional[str] = None, transform: Optional[_transform] = None) -> None:
        self.dataset: pd.DataFrame = pd.read_csv(data_path, sep= ',')
        if target:
            self.X: pd.DataFrame = self.dataset.drop(target, axis= 1)
            self.y: pd.DataFrame = self.dataset[target]
        if transform:
            self.transform = transform
            self.dataset = self.transform(self.dataset)
    

    def __len__(self) -> tuple[int, int]:
        return self.dataset.shape


    def __getitem__(self, index: int) -> pd.DataFrame:
        return super().__getitem__(index)



#@: ----  Class Data Analysis ----
class KidneyAnalysis:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })


    __str__ = __repr__


    @classmethod
    def characteristics(cls, dataset: pd.DataFrame, 
                             info: Optional[bool] = False, 
                             summary: Optional[bool] = False ) -> Generator[str|Any, None, None]:
        yield f"Dataset Shape: {dataset.shape}"
        yield f"Dataset features: {dataset.columns.values}"
        if info:
            yield f"Dataset Info: {dataset.info()}"
        yield f"Dataset types: {dataset.dtypes}"
        if summary:
            yield dataset.describe().T
        

    @classmethod
    def unique_values(cls, dataset: pd.DataFrame) -> Generator[str, None, None]:
        for i in dataset.columns:
            yield f"Unique value in {i} : {dataset[i].unique()}"


    @classmethod
    def data_typos_cleaning(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        for i in range(dataset.shape[0]):
            if dataset.iloc[i,25]=='ckd\t':
                dataset.iloc[i,25]='ckd'
            if dataset.iloc[i,20] in [' yes','\tyes']:
                dataset.iloc[i,20]='yes'
            if dataset.iloc[i,20]=='\tno':
                dataset.iloc[i,20]='no'
            if dataset.iloc[i,21]=='\tno':
                dataset.iloc[i,21]='no'
            if dataset.iloc[i,16]=='\t?':
                dataset.iloc[i,16]=np.nan
            if dataset.iloc[i,16]=='\t43':
                dataset.iloc[i,16]='43'
            if dataset.iloc[i,17]=='\t?':
                dataset.iloc[i,17]=np.nan
            if dataset.iloc[i,17]=='\t6200':
                dataset.iloc[i,17]= '6200'
            if dataset.iloc[i,17]=='\t8400':
                dataset.iloc[i,17]= '6200'
            if dataset.iloc[i,18]=='\t?':
                dataset.iloc[i,18]=np.nan
            if dataset.iloc[i,25]=='ckd':
                dataset.iloc[i,25]='yes'
            if dataset.iloc[i,25]=='notckd':
                dataset.iloc[i,25]='no'
        return dataset


    @classmethod
    def chance_feature_names(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset.drop('id', axis= 1, inplace= True)
        feature_names: list[str] = [
            'Age (yrs)', 'Blood Pressure (mm/Hg)', 'Specific Gravity', 
            'Albumin',   'Sugar',  'Red Blood Cells',
            'Pus Cells', 'Pus Cell Clumps', 'Bacteria', 
            'Blood Glucose Random (mgs/dL)', 'Blood Urea (mgs/dL)',
            'Serum Creatinine (mgs/dL)', 'Sodium (mEq/L)', 'Potassium (mEq/L)', 
            'Hemoglobin (gms)', 'Packed Cell Volume',
            'White Blood Cells (cells/cmm)', 'Red Blood Cells (millions/cmm)', 
            'Hypertension', 'Diabetes Mellitus',
            'Coronary Artery Disease', 'Appetite', 'Pedal Edema', 
            'Anemia', 'Chronic Kidney Disease']
        dataset.columns = feature_names
        return dataset


    @classmethod
    def numerics_features(cls, dataset: pd.DataFrame) -> list[str]:
        mistyped: list[str] = [
            'Packed Cell Volume','White Blood Cells (cells/cmm)','Red Blood Cells (millions/cmm)'
        ]
        numeric: list[str] = []
        for col in mistyped:
            dataset[col] = dataset[col].astype('float')

        for i in dataset.columns:
            if dataset[i].dtype=='float64':
                numeric.append(i)
        numeric = numeric + mistyped
        return numeric


    @classmethod
    def categorical_features(cls, dataset: pd.DataFrame) -> list[str]:
        categoricals: list[str] = []
        for col in dataset.columns:
            if not col in cls.numerics_features(dataset):
                categoricals.append(col)
        categoricals.remove('Chronic Kidney Disease')
        return categoricals
        
    
    

#@: Kidney Data Visualization
class KidneyVisualization:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'], 
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    __str__ = __repr__


    @classmethod
    def numerical_features_distribution(cls, dataset: pd.DataFrame, numeric: list[str]) -> _plot:
        n_rows, n_cols = (7,2)
        figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols,figsize=(20, 50))
        figure.suptitle('\n\nDistributions of Numerical Features', fontsize=60)
        
        for index, column in enumerate(numeric):
            i,j = (index // n_cols), (index % n_cols)
            miss_perc = "%.2f"%(100*(1-(dataset[column].dropna().shape[0])/dataset.shape[0]))
            collabel = column+"\n({}% is missing)".format(miss_perc)
            fig = sns.distplot(dataset[column], color="g", label=collabel, norm_hist=True,
                                ax=axes[i,j], kde_kws={"lw":4})
            fig = fig.legend(loc='best', fontsize=18)
            axes[i,j].set_ylabel("Probability Density",fontsize='medium')
            axes[i,j].set_xlabel(None)
        plt.show()

        


#@: ---- Driver code ---- 
if __name__.__contains__('__main__'):
    kidney: object = KidneyDataset('C:\\Users\\Lenovo\\OneDrive\\Desktop\\__Desktop\\Kidney\\kidney_disease.csv')
    #print(kidney.dataset)
    numerics_features: list[str] = KidneyAnalysis.numerics_features(dataset= kidney.dataset)
    KidneyVisualization.numerical_features_distribution(dataset= kidney.dataset, numeric= numerics_features)