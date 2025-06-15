from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta #to calculate relative dates
import os

#Import your existing functions
from pathlib import Path
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import mlflow

def setup():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    return df

def create_X(df, dv=None):
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

def train_and_log_model(**context):
    setup()
    task_instance = context['task_instance']
    dates = task_instance.xcom_pull(task_ids='calculate_dates')

    # Fallback to manual variables if needed
    if not dates:
        train_year = int(Variable.get("training_year", default_var=datetime.now().year))
        train_month = int(Variable.get("training_month", default_var=datetime.now().month))
        val_year = train_year if train_month < 12 else train_year + 1
        val_month = train_month + 1 if train_month < 12 else 1
    else:
        train_year = dates['train_year']
        train_month = dates['train_month']
        val_year = dates['val_year']
        val_month = dates['val_month']

    df_train = read_dataframe(train_year, train_month)
    df_val = read_dataframe(val_year, val_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    y_train = df_train['duration'].values
    y_val = df_val['duration'].values

    with mlflow.start_run() as run:
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"MLflow run_id: {run.info.run_id}")

def prepare_dates(**context):
        """Calculate training and validation dates based on current execution date"""
        logical_date = context['logical_date']
        
        # Training data: 4 months ago, because 2 months ago isn't available yet
        train_date = logical_date - relativedelta(months=4)
        train_year = train_date.year
        train_month = train_date.month
        
        # Validation data: 3 months ago  
        val_date = logical_date - relativedelta(months=3)
        val_year = val_date.year
        val_month = val_date.month
        
        print(f"Execution date: {logical_date.strftime('%Y-%m')}")
        print(f"Training data: {train_year}-{train_month:02d}")
        print(f"Validation data: {val_year}-{val_month:02d}")
        
        return {
            'train_year': train_year,
            'train_month': train_month,
            'val_year': val_year,
            'val_month': val_month
        }

default_args = {
    'owner': 'phoebe',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Define the DAG
dag = DAG (
    default_args=default_args,
    dag_id='nyc_taxi_train_pipeline',
    start_date=datetime(2025, 6, 8),
    schedule='@monthly',  # Trigger manually
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'nyc']
)
    

calculate_dates_task = PythonOperator(
    task_id='calculate_dates',
    python_callable=prepare_dates,
    dag=dag,
)

run_training = PythonOperator(
    task_id='train_model_task',
    python_callable=train_and_log_model,
)

calculate_dates_task >> run_training