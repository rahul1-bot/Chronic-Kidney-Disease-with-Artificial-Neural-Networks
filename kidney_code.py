from __future__ import annotations

__authors__: list[str] = ['Rahul Sawhney', 'Leah Khan']

__doc__: str = r'''
    >>> Paper Topic: 
        Classification of Chronic Kidney Disease with Artificial Neural Networks
    
    >>> Paper Abstract:
        Chronic Kidney Disease or CKD is one of the most prevalent disease which influence humans on a larger scale 
        and proves to be fatal as it remains dormant unless irreversible damages have been made to the kidney of an 
        individual. Progression of CKD is related to variety of great complications, including increased incidence 
        of various disorders, anemia, hyperlipidemia, nerve damage, pregnancy complications and even complete kidney
        failure. Millions of people die from this disease every year. Diagnosing CKD is a cumbersome task as there 
        are no major symptom that can be used as a benchmark to detect the disease. In cases when diagnosis persists,
        some results may be interpreted incorrectly. This paper proposes a Multi- Layered Perceptron Classifier that
        uses deep neural network in order to predict whether a patient has CKD or not. The model is trained on a dataset
        of about four hundred patients and considers diverse signs and symptoms which includes blood pressure, age, 
        sugar level, red blood cell count, etc. The experimental results display that the proposed model can perform 
        classification with the testing accuracy of 100 %. The aim is to help introduce Deep Learning methods in learning
        from the dataset attribute reports and detect CKD correctly to a large extent.

'''
import os, warnings, time, copy
warnings.filterwarnings('ignore')
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.impute import KNNImputer

import torch
import torch.nn as nn
import torch.nn.functional as F


#@: Data Analysis Step 
class KidneyAnalysis:
    def rename_columns(self, dataset: pd.DataFrame) -> pd.DataFrame:
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



    def data_characteristics(self, dataset: pd.DataFrame, info: Optional[bool] = False) -> Generator:
        yield f'Dataset Shape: {dataset.shape}'
        yield f'Dataset features: {dataset.columns.values}'
        if info:
            yield f'Dataset Info: {dataset.info()}'
        yield f'Dataset types: {dataset.dtypes}'
        yield dataset.describe().T
        yield f'Null Values in Dataset: {dataset.isnull().sum()}'



    def unique_values(self, dataset: pd.DataFrame) -> Generator:
        for column in dataset.columns:
            yield f'Unique Value in {column} :\n{dataset[column].unique()}'

    

    def target_distribution(self, dataset: pd.DataFrame, target: str) -> 'plot':
        sns.countplot(x= target, data= dataset)
        plt.xlabel(target)
        plt.ylabel('Count')
        plt.title('Target Classification Distribution')
        plt.show()



    def histoplots(cls, dataset: pd.DataFrame, numeric_features: list[str]) -> 'plot':
        dataset.hist(numeric_features, figsize = (15, 8))
        plt.show()



#@: Data Preprocess Step
class KidneyPreprocess:
    def replace_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
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



    def object_to_str(self, dataset: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        for col in cols:
            dataset[col] = dataset[col].astype(str)
        return dataset



    def encode_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
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



    def impute_NANs(self, dataset: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
        y: pd.Series = dataset[target]
        imputer = KNNImputer(n_neighbors= 5, weights= 'uniform', metric= 'nan_euclidean')
        impute_cols: list[str] = list(
            set(dataset.columns) - set(['classification'])
        )
        imputer.fit(dataset[impute_cols])
        X: pd.DataFrame = pd.DataFrame(
            data= imputer.transform(dataset[impute_cols]),
            columns= impute_cols
        )
        return X, y
    


    def missingValues_percentage(self, X: pd.DataFrame, y: pd.Series) -> Generator[pd.Series, None, None]:
        yield round(
            (X.isnull().sum() * 100/ len(X)), 2
        ).sort_values(ascending=False)
        yield round(
            (y.isnull().sum() * 100/ len(y)), 2
        ).sort_values(ascending=False)




#@: Machine-Learning Step
class TraditionalModels:
    def metric(self, y_test: pd.Seris, y_pred: pd.series) -> 'plot':
        cm = confusion_matrix(y_test, y_pred)
        group_names: list[str] = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
        group_counts: list = [
            '{0:0.0f}'.format(value) for value in cm.flatten()
        ]
        group_percentages: list = [
            "{0:.2%}".format(value) 
            for value in cm.flatten() / np.sum(cm)
        ]
        labels: list = [
            f"{v1}\n{v2}\n{v3}" 
            for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
        ]

        labels: np.ndarray = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cm, annot= labels, fmt= '', cmap= 'Blues')
        plt.show()
        return classification_report(y_test, y_pred)




    def logistic_regression(self, x_train: pd.Series, y_train: pd.Series, 
                                                      x_test: pd.Series, 
                                                      y_test: pd.Series) -> Generator:
        c_space = np.logspace(-5, 8, 15)
        params: dict[str, np.ndarray] = {
            'C': c_space
        }
        model = LogisticRegression()
        model = GridSearchCV(model, param_grid= params, cv= 5)
        model.fit(x_train, y_train)
        yield f'Tunned Params : {model.best_params_}'
        yield f'Best Score in Training Dataset : {model.best_score_}'

        y_pred = model.predict(x_test)
        accuracy: float = accuracy_score(y_test, y_pred)
        yield f'Accuracy of the Model : {accuracy}'
        yield {'Accuracy': accuracy}
        yield self.metric(y_test, y_pred)




    def support_vector_Machines(self, x_train: pd.Series, y_train: pd.Series, 
                                                          x_test: pd.Series, 
                                                          y_test: pd.Series) -> Generator:
        params: dict[str, list] = {
            'C'     : [0.1, 1, 10, 100, 1000],
            'gamma' : [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }
        model: _model = SVC()
        model: _model = GridSearchCV(estimator= model, param_grid= params, cv= 5)
        model.fit(x_train, y_train)
        yield f'Tunned Params : {model.best_params_}'
        yield f'Best Score in Training Dataset : {model.best_score_}'

        y_pred = model.predict(x_test)
        accuracy: float = accuracy_score(y_test, y_pred)
        yield f'Accuracy of the Model : {accuracy}'
        yield {'Accuracy': accuracy}
        yield self.metric(y_test, y_pred)



    
    def decision_trees(self, x_train: pd.Series, y_train: pd.Series, 
                                                 x_test: pd.Series, 
                                                 y_test: pd.Series) -> Generator:
        params: dict[str, list] = {
            'max_depth': [4, 6, 8, 10, 12],
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [2, 10, 20, 30, 40],
            'max_features': [0.2, 0.4, 0.6, 0.8, 1],
            'max_leaf_nodes': [8, 16, 32, 64, 128],
            'class_weight': [{0: 1, 1: 1}, 
                             {0: 1, 1: 2}, 
                             {0: 1, 1: 3}, 
                             {0: 1, 1: 4}, 
                             {0: 1, 1: 5}]
        }

        model: _model = RandomizedSearchCV(estimator=  DecisionTreeClassifier(), 
                                           param_distributions= params,
                                           scoring= 'f1',
                                           random_state= 1,
                                           n_iter= 20)
        model = model.fit(x_train, y_train)
        yield f'Tunned Params : {model.best_params_}'
        yield f'Best Score in Training Dataset : {model.best_score_}'
        
        y_pred = model.predict(x_test)
        accuracy: float = accuracy_score(y_test, y_pred)
        yield f'Accuracy of the Model : {accuracy}'
        yield {'Accuracy': accuracy}
        yield self.metric(y_test, y_pred)




    def random_forest(self, x_train: pd.Series, y_train: pd.Series, 
                                                x_test: pd.Series, 
                                                y_test: pd.Series) -> Generator:
        param: dict[str, np.ndarray] = {
            'n_estimators': np.arange(2, 300, 2),
            'max_depth': np.arange(1, 28, 1),
            'min_samples_split': np.arange(1,150,1),
            'min_samples_leaf': np.arange(1,60,1),
            'max_leaf_nodes': np.arange(2,60,1),
        }

        model: _model = RandomizedSearchCV(estimator= RandomForestClassifier(),
                                           param_distributions= param,
                                           scoring= 'f1',
                                           random_state= 1,
                                           n_iter= 20)
        
        model: _model = model.fit(x_train, y_train)
        yield f'Tunned Params : {model.best_params_}'
        yield f'Best Score in Training Dataset : {model.best_score_}'
        
        y_pred = model.predict(x_test)
        accuracy: float = accuracy_score(y_test, y_pred)
        yield f'Accuracy of the Model : {accuracy}'
        yield {'Accuracy': accuracy}
        yield self.metric(y_test, y_pred)





#@: ----- Deep-Learning Step ------ 
class KidneyDataset(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X: np.ndarray = X.values
        self.y: np.ndarray = y.values
    

    def __len__(self) -> int:
        return len(self.X)


    def __getitem__(self, index: int) -> tuple:
        return self.X[index], self.y[index]



class ForwardNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(ForwardNet, self).__init__()
        layers: dict[str, object] = {
            'linear_1': nn.Linear(input_dim, hidden_dim),
            'relu_1': nn.ReLU(),

            'linear_2': nn.Linear(hidden_dim, hidden_dim),
            'relu_2': nn.ReLU(),

            'linear_3': nn.Linear(hidden_dim, hidden_dim),
            'relu_3': nn.ReLU(),

            'linear_4': nn.Linear(hidden_dim, hidden_dim),
            'relu_4': nn.ReLU(),

            'linear_5': nn.Linear(hidden_dim, hidden_dim),
            'relu_5': nn.ReLU(),

            'linear_6': nn.Linear(hidden_dim, output_dim)           
        }
        self.block = nn.Sequential(*layers.values())
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x




class Model():
    def __init__(self, net: 'model', criterion: object, 
                                     optimizer: object, 
                                     num_epochs: int, 
                                     dataloaders: dict[int, object], 
                                     dataset_sizes: dict[str, int], 
                                     device: torch.device) -> None:
        super(Model, self).__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.device = device
    


    def train_epoch(self) -> tuple[float, float]:
        self.net.train()
        total_correct: int = 0
        total_loss: float = 0.0
        total_examples: int = 0
        
        for X, y in self.dataloaders['train']:
            X = X.to(self.device)
            y = y.to(self.device) 
            y_hat = self.net(X.float())
            loss = self.criterion(y_hat, y.long())
            
            #@: back propogation
            self.net.zero_grad()
            loss.backward()
            self.optimizer.step()
            #@: end back propogation

            total_examples += y.size(0)
            total_loss += loss.item()
            total_correct += (torch.argmax(y_hat, 1) == y).sum().item()

        return total_loss / len(self.dataloaders['train']), total_correct / total_examples 

        

    def test_epoch(self) -> tuple[float, float]:
        self.net.eval()
        total_correct: int = 0
        total_examples: int = 0
        total_loss: float = 0.0
        
        for X, y in self.dataloaders['test']:
            X = X.to(self.device)
            y = y.to(self.device)
            y_hat = self.net(X.float())
            loss = self.criterion(y_hat, y.long())

            total_examples += y.size(0)
            total_loss += loss.item()
            total_correct += (torch.argmax(y_hat, 1) == y).sum().item()

        return total_loss / len(self.dataloaders['test']), total_correct / total_examples




    def fit(self) -> 'text':
        total_train_accuracy: list[float] = []
        total_test_accuracy: list[float] = []
        max_train_acc: float = 0.0
        max_val_acc: float = 0.0

        for epoch in range(self.num_epochs):
            print("------ Epoch {:02d} ------".format(epoch))
            
            loss, train_acc = self.train_epoch()
            if train_acc > max_train_acc:
                max_train_acc = train_acc 
            total_train_accuracy.append(train_acc)
            print("Train Loss: {:.04f}, Accuracy: {:.04f}".format(loss, train_acc))
            
            
            loss, test_acc = self.test_epoch()
            if test_acc > max_val_acc:
                max_val_acc = test_acc
            total_test_accuracy.append(test_acc) 
            print("Test  Loss: {:.04f}, Accuracy: {:.04f}".format(loss, test_acc))


        print(f'Training Accuracy: {max_train_acc}')
        print(f'Testing Accuracy: {max_val_acc}')
        



#@: Driver code
if __name__.__contains__('__main__'):
    data_path: 'path' = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\_!\\kidney_disease.csv'
    kidney_data = pd.read_csv(data_path)
    
    kidney_data = KidneyAnalysis().rename_columns(kidney_data)
    
    data_characteristic = KidneyAnalysis().data_characteristics(kidney_data)
    while True:
        try: 
            print('-' * 100)
            print(data_characteristic.__next__())
        except StopIteration: break
    
    KidneyAnalysis().target_distribution(kidney_data, 'classification')
    KidneyAnalysis().histoplots(kidney_data, numeric_features= [
                                                'age',
                                                'blood_pressure', 
                                                'blood_glucose_random', 
                                                'sodium', 
                                                'potassium', 
                                                'packed_cell_volume', 
                                                'red_blood_cell_count'
    ])

    kidney_data = KidneyPreprocess().replace_values(kidney_data)
    kidney_data = KidneyPreprocess().object_to_str(kidney_data, cols= [
                                                'red_blood_cells', 
                                                'pus_cell', 
                                                'pus_cell_clumps', 
                                                'bacteria', 
                                                'hypertension', 
                                                'diabetes_mellitus',
                                                'coronary_artery_disease', 
                                                'pedal_edema', 
                                                'anemia', 
                                                'appetite'
    ])

    kidney_data = KidneyPreprocess().encode_features(kidney_data)
    X, y = KidneyPreprocess().impute_NANs(kidney_data, target= 'classification')
    X = X.drop('id', axis= 1)
    print(len(X.keys()))

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 7)

    accuracies: list[float] = []
    
    logistic_regression = TraditionalModels().logistic_regression(x_train, y_train, x_test, y_test)
    while True:
        try:
            item = logistic_regression.__next__()
            if isinstance(item, dict):
                accuracies.append(item['Accuracy'])
            else:
                print(item)
        except StopIteration: break
    

    support_vector_machines = TraditionalModels().support_vector_Machines(x_train, y_train, x_test, y_test)
    while True:
        try:
            item = support_vector_machines.__next__()
            if isinstance(item, dict):
                accuracies.append(item['Accuracy'])
            else:
                print(item)
        except StopIteration: break


    decision_trees = TraditionalModels().decision_trees(x_train, y_train, x_test, y_test)
    while True:
        try:
            item = decision_trees.__next__()
            if isinstance(item, dict):
                accuracies.append(item['Accuracy'])
            else:
                print(item)
        except StopIteration: break

    
    random_forest = TraditionalModels().random_forest(x_train, y_train, x_test, y_test)
    while True:
        try:
            item = random_forest.__next__()
            if isinstance(item, dict):
                accuracies.append(item['Accuracy'])
            else:
                print(item)
        except StopIteration: break

    
    train_data = KidneyDataset(X= x_train, y= y_train)
    test_data = KidneyDataset(X= x_test, y= y_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size= 16, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= 16)

    dataset_sizes: dict[str, int] = {
        'train': len(train_data),
        'test': len(test_data)
    }

    dataloaders: dict[str, object] = {
        'train': train_loader,
        'test': test_loader
    }
    device = 'cpu'

    model = ForwardNet(input_dim= 24, hidden_dim= 64, output_dim= 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
    

    kidney_classifier_ann = Model(
        net= model, 
        criterion= criterion, 
        optimizer= optimizer,
        num_epochs= 250, 
        dataloaders= dataloaders, 
        dataset_sizes= dataset_sizes, 
        device= device
    )
    kidney_classifier_ann.fit()
    