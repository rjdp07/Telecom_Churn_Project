import pandas as pd
import os
from churnPrediction import logger
from xgboost import XGBClassifier
import joblib
from src.churnPrediction.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        X_train = pd.read_csv(self.config.Xtrain_data_path)
        X_test  = pd.read_csv(self.config.Xtest_data_path)
        y_train  = pd.read_csv(self.config.ytrain_data_path)
        y_test  = pd.read_csv(self.config.ytest_data_path)

        xgb_model = XGBClassifier(colsample_bytree = self.config.colsample_bytree,
                                  gamma = self.config.gamma,
                                  learning_rate = self.config.learning_rate,
                                  max_depth = self.config.max_depth,
                                  min_child_weight = self.config.min_child_weight)
        xgb_model.fit(X_train,y_train)
        joblib.dump(xgb_model, os.path.join(self.config.root_dir, self.config.model_name))