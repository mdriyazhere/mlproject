## Modelling
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
##from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
##from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
##from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.ensemble import AdaBoostRegressor ##AdaBoostClassifier
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression,ElasticNet,Ridge,Lasso
from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
import os
import sys
import pandas as pd
import numpy as np
from src.pipeline.utils import save_object,evaluate_models
from dataclasses import dataclass
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

@dataclass
class ModelTrainerConfig:
        trained_model_file=os.path.join('artifacts',"model.pkl")
class ModelTrainer:
        def __init__(self):
            self.model_trained_config=ModelTrainerConfig()
        def initiate_model_trainer(self,train_array,test_array,preprocessor_path):
                try:
                    logging.info("splitting training and test input data")
                    x_train,y_train,x_test,y_test=(
                      train_array[:,:-1],
                      train_array[:,-1],
                      test_array[:,:-1],
                      test_array[:,-1])
                    models={
                        "LinearRegression":LinearRegression(),
                        "logisticRegression": LogisticRegression(),
                        "Decision Tree" :DecisionTreeRegressor(),
                        "Lasso":Lasso(),
                        "Ridge":Ridge(),
                        "svm":SVR(),
                        "catboost":CatBoostRegressor(verbose=False),
                        "AdaBoostRegressor":AdaBoostRegressor(),
                        "KNeighborsRegressor":KNeighborsRegressor(),
                        "RandomForestRegressor":RandomForestRegressor(),
                        "XGBRegressor":XGBRegressor(),
                        "GradientBoostRegressor":GradientBoostingRegressor()
}
                    model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
                    ## to get best model score from dict
                    best_model_score=max(sorted(model_report.values()))
                    ## to get best model name from dict
                    best_model_name=list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
                    ]
                    best_model=models[best_model_name]

                    if best_model_score<0.6:
                        raise CustomException("no best model found")
                    logging.info(f"best model and best model score found for training and testing dataset")
                    
                    save_object(
                        file_path=self.model_trained_config.trained_model_file,
                        obj=best_model
                    )
                    predicted=best_model.predict(x_test)
                    r2_square=r2_score(y_test,predicted)
                    return r2_square
                
                except Exception as e:
                    raise CustomException(e,sys)
                    
