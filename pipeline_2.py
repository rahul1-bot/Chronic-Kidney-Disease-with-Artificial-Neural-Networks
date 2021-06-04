from __future__ import annotations
from pipeline_1 import *
from typing import NewType, Any

# Scripting Ml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


#@: Pipeline_2: Machine_Learning
    #@: class Traditional_Models
    #       : __repr__                          -> str(dict[str, str])
    #       : __str__                           -> str(dict[str, str])
    #       : metric()                          -> _plot
    #       : LogisticRegression()              -> Generator[Any, None, None]
    #       : SupportVectorMachines()           -> Generator[Any, None, None]
    #       : DecisionTrees()                   -> Generator[Any, None, None]
    #       : RandomForest()                    -> Generator[Any, None, None]
    #    
    #@: if __name__.__contains__('__main__')   



# torch typing scripts
_path   = NewType('_path', Any)
_loader = NewType('_loader', Any)
_model  = NewType('_model', Any)
_loss   = NewType('_loss', Any)
_optimizer = NewType('_optimizer', Any)
_transform = NewType('_transform', Any)
_text = NewType('_text', Any)
_imputer = NewType('_Imputer', Any)


#@: class Traditional_Models
class Traditional_Models:
    def __repr__(self) -> str(dict[str, str]):
        return str({
            item: value for item, value in zip(['Module', 'Name', 'ObjectID'],
                                               [self.__module__, type(self).__name__, hex(id(self))])
        })

    __str__ = __repr__



    @classmethod
    def metric(cls, y_test: pd.Series, y_pred: pd.Series) -> _plot:
        cm: _plot = confusion_matrix(y_test, y_pred)
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




    @classmethod
    def LogisticRegression(cls, x_train: pd.DataFrame, y_train: pd.Series, 
                                x_test: pd.DataFrame,  y_test: pd.Series) -> Generator[Any, None, None]:
        c_space: np.ndarray = np.logspace(-5, 8, 15)
        params: dict[str, np.ndarray] = {
            'C': c_space
        }
        model: _model = LogisticRegression()
        model: _model = GridSearchCV(model, param_grid= params, cv= 5)
        model.fit(x_train, y_train)
        yield f'Tunned Params : {model.best_params_}'
        yield f'Best Score in Training Dataset : {model.best_score_}'

        y_pred = model.predict(x_test)
        accuracy: float = accuracy_score(y_test, y_pred)
        yield f'Accuracy of the Model : {accuracy}'
        yield {'Accuracy': accuracy}
        yield cls.metric(y_test, y_pred)
    


    @classmethod
    def SupportVectorMachines(cls, x_train: pd.DataFrame, y_train: pd.Series, 
                                    x_test: pd.DataFrame, y_test: pd.Series) -> Generator[Any, None, None]:
        
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
        yield cls.metric(y_test, y_pred)
        


    @classmethod
    def DecisionTrees(cls, x_train: pd.DataFrame, y_train: pd.Series,
                           x_test: pd.DataFrame,  y_test: pd.Series) -> Generator[Any, None, None]:
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
        yield cls.metric(y_test, y_pred)
        



    @classmethod
    def RandomForest(cls, x_train: pd.DataFrame, y_train: pd.Series, 
                          x_test: pd.DataFrame, y_test: pd.Series) -> Generator[Any, None, None]:
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
        yield cls.metric(y_test, y_pred)






# Driver Code
if __name__.__contains__('__main__'):
    kidney: object = KidneyDataset(path= 'C:\\Users\\Lenovo\\OneDrive\\Desktop\\_Kidney\\kidney_disease.csv')
    #print(len(kidney))
    kidney.dataset: pd.DataFrame = KidneyAnalysis.rename_columns(kidney.dataset)
    #print(kidney.dataset['diabetes_mellitus'].unique())

    data_characteristics: Generator = KidneyAnalysis.data_characteristics(kidney.dataset, True, True)
    
    '''
    while True:
        try:
            print('-'*100)
            print(data_characteristics.__next__())
        except StopIteration: break
    '''

    #target_count: Generator = KidneyAnalysis.target_classification_count(kidney.dataset, 'classification')
    '''
    while True:
        try:
            print('-'*100)
            print(target_count.__next__())
        except StopIteration: break
    '''
    #kidney.dataset: pd.DataFrame = KidneyAnalysis.replace_incorrect_values(kidney.dataset)
    #print(kidney.dataset['classification'].unique())
    #KidneyAnalysis.target_distribution(kidney.dataset, 'classification')
    #KidneyAnalysis.histograms_numeric_features(kidney.dataset, ['age','blood_pressure', 'blood_glucose_random', 'blood_glucose_random', 
     #                                                           'blood_glucose_random', 'sodium', 'potassium',  
      #                                                          'packed_cell_volume', 'packed_cell_volume', 'red_blood_cell_count'])


    kidney.dataset: pd.DataFrame = KidneyPreprocess.replace_values(kidney.dataset)
    kidney.dataset: pd.DataFrame = KidneyPreprocess.change_object_to_str(kidney.dataset, ['red_blood_cells', 'pus_cell', 'pus_cell_clumps',
                                                                                          'bacteria', 'hypertension', 'diabetes_mellitus',
                                                                                          'coronary_artery_disease', 'pedal_edema', 'anemia', 'appetite'])
    kidney.dataset: pd.DataFrame = KidneyPreprocess.encode_features(kidney.dataset)
    #print(kidney.dataset['red_blood_cells'])


    X, y = KidneyPreprocess.impute_nan_values(kidney.dataset, 'classification')
    #print(kidney.dataset['classification'])
    #print(X)
    X: pd.DataFrame = X.drop('id', axis= 1)
    #print(X)
    #print(y.unique())
    #X.info()

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.20, random_state= 7)
    
    logistic_classifier: Generator = Traditional_Models.LogisticRegression(x_train, y_train, x_test, y_test)

    accuracies: list[float] = []
    '''
    while True:
        try:
            item = logistic_classifier.__next__()
            if isinstance(item, dict):
                accuracies.append(item['Accuracy'])
            else:
                print(item)
        except StopIteration: break
    
    print(accuracies)
    '''
    '''


    # NN Model 
    # print('hemllo')

    x_train_tensor: torch.Tensor = torch.FloatTensor(x_train.values)
    x_test_tensor: torch.Tensor = torch.FloatTensor(x_test.values)
    y_train_tensor: torch.Tensor = torch.LongTensor(y_train.values)
    y_test_tensor: torch.Tensor = torch.LongTensor(y_test.values)

    # print(type(x_train_tensor))
    # print(x_train_tensor)

    train_loader: _loader = DataLoader(dataset= x_train_tensor, 
                                       batch_size= 60, 
                                       shuffle= True)

    test_loader: _loader = DataLoader(dataset= x_test_tensor, 
                                      batch_size= 60, 
                                      shuffle= True)
    

                

    
        
    '''

    random_forest: Generator = Traditional_Models.RandomForest(x_train, y_train, 
                                                           x_test, y_test)
    while True:
        try:
            item = random_forest.__next__()
            #if isinstance(item, dict):
                #accuracies.append(item['Accuracy'])
            #else:
            print(item)
        except StopIteration: break