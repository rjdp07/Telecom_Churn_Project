import os
import pandas as pd
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from churnPrediction.entity.config_entity import ModelEvaluationConfig
from churnPrediction.utils.common import read_yaml, create_directories, save_json
from pathlib import Path
from churnPrediction import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy =accuracy_score(actual,pred)
        return accuracy
    
    def log_into_mlfow(self):

        X_test = pd.read_csv(self.config.Xtest_data_path)
        y_test = pd.read_csv(self.config.ytest_data_path)

        model = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        print(tracking_url_type_store)
        mlflow.end_run()

        with mlflow.start_run():
            predicted_qualities = model.predict(X_test)

            model_accuracy = self.eval_metrics(y_test,predicted_qualities)

            scores = {"accuracy":model_accuracy}
            save_json(path = Path(self.config.metric_file_name), data = scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("accuracy",model_accuracy)
            
            #Model registry does not work with file store
            if tracking_url_type_store != "file":
                #Register the model
                #There are other ways to use model registry, which depends on the use case,
                #please refer to the doc for more information
                mlflow.sklearn.log_model(model, "model", registered_model_name="XGB")
                
            else:
                
                mlflow.sklearn.log_model(model, "model")
                logger.info("run here???")
