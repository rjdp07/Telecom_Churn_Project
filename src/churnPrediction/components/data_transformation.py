import os
from churnPrediction import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from churnPrediction.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import numpy as np



class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
        self.raw_data = pd.read_excel(self.config.data_path)

    def ColumnConvert(self):

        self.raw_data["SeniorCitizen"] = self.raw_data["SeniorCitizen"].astype('category')

        self.raw_data["TotalCharges"] = np.where(self.raw_data["TotalCharges"] == " ","0",self.raw_data["TotalCharges"])
        self.raw_data["TotalCharges"] = self.raw_data["TotalCharges"].astype("float")
        logger.info("Converting Columns to Correct Data Format")

    def XYDataSets(self):
        self.X = self.raw_data.drop(columns=['customerID','Churn'],axis = 1)
        self.y = self.raw_data['Churn']

        logger.info("X and Y Data object Created")


    def ColumnPreProcess(self):
        num_feature = self.X.select_dtypes(exclude='object').columns
        cat_feature = self.X.select_dtypes(include = 'object').columns

        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder()

        preprocessor = ColumnTransformer(
            [('OneHotEncoder',cat_transformer, cat_feature),
            ('StandardScaler', num_transformer, num_feature)]
        )
        
        self.X = preprocessor.fit_transform(self.X)

        logger.info("Numeric Columns Scaled || Categorical Columns Dummy Encoded")


    def ClassBalancing(self):
        sm = SMOTE(random_state = 42)
        self.X_smote, self.y_smote = sm.fit_resample(self.X, self.y)

        logger.info("SMOTE Class Balancing Applied")

    


    def train_test_splitting(self):

        data = self.raw_data
        X_train, X_test, y_train, y_test = train_test_split(self.X_smote, self.y_smote, test_size = 0.3, random_state=720)

        #X_train.to_csv(os.path.join(self.config.root_dir,"X_train.csv"), index = False)
        np.savetxt(os.path.join(self.config.root_dir,"X_train.csv"), X_train, delimiter=",", fmt='%s')
        #X_test.to_csv(os.path.join(self.config.root.dir,"X_test.csv"), index = False)
        np.savetxt(os.path.join(self.config.root_dir,"X_test.csv"), X_test, delimiter=",", fmt='%s')
        #y_train.to_csv(os.path.join(self.config.root.dir,"y_train.csv"), index = False)
        np.savetxt(os.path.join(self.config.root_dir,"y_train.csv"), y_train, delimiter=",", fmt='%s')
        #y_test.to_csv(os.path.join(self.config.root.dir,"y_test.csv"), index = False)
        np.savetxt(os.path.join(self.config.root_dir,"y_test.csv"), y_test, delimiter=",", fmt='%s')

        logger.info("Splitted data into training and test sets")
        logger.info(X_train.shape)
        logger.info(X_test.shape)

        print(X_train.shape)
        print(X_test.shape)