# Telecom_Churn_Project
An End-to-End ML Project for Predicting Churn on a Telecom Company

## Workflows

1. Update config.yaml
2. Update schema.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
8. Update the main.py
9. Update the dvc.yaml


# How to run?

### STEPS:

Clone the repository


```bash
https://github.com/rjdp07/Telecom_Churn_Project
```

### STEP 01 - Create a conda environment after opening the repository

```bash
conda create -n ML_Churn_Pred python=3.8 -y
```

```bash
conda activate ML_Churn_Pred
```


### STEP 02 - install the requirements
```bash
pip install -r requirements.txt
```

```bash
python app.py
```

Now,
```bash
open up your local host and port
```

## MLflow
[Documentation](https://mlflow.org/docs/latest/index.html)

##### cmd
- mlflow ui


### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/rjdp07/Telecom_Churn_Project.mlflow \
MLFLOW_TRACKING_USERNAME=rjdp07 \
MLFLOW_TRACKING_PASSWORD=0a9816f8c59f4bd161a99945ec3bb97d3faa61cf \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/rjdp07/Telecom_Churn_Project.mlflow

export MLFLOW_TRACKING_USERNAME=rjdp07

export MLFLOW_TRACKING_PASSWORD=0a9816f8c59f4bd161a99945ec3bb97d3faa61cf

```





