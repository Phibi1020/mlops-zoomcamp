import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow
from sklearn.feature_extraction import DictVectorizer

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def setup():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    mlflow.sklearn.autolog()
    dv = DictVectorizer()
    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    for model_class in (RandomForestRegressor,):
    
        with mlflow.start_run():
    
            mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.csv")
            mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.csv")
            
            mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
    
            mlmodel = model_class()
            mlmodel.fit(X_train, y_train)
    
            y_pred = mlmodel.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mlflow.log_metric("rmse", rmse)
        



if __name__ == '__main__':
    setup()
    run_train()
