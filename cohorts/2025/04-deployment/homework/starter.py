#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def read_data(year, month, categorical):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def create_X(df, categorical):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    return X_val
    

def run(year, month):
    categorical = ['PULocationID', 'DOLocationID']
    df = read_data(year=year, month=month, categorical = categorical)
    X_val = create_X(df=df, categorical = categorical)
    y_pred = model.predict(X_val)
    return y_pred


app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    date = request.get_json()

    y_pred = run(year=date["year"], month=date["month"])
    mean = np.mean(y_pred)

    result = {
        'Mean': mean
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
