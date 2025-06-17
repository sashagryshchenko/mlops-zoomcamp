#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import click
import pandas as pd
import numpy as np

# Globals
CATEGORICAL = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[CATEGORICAL] = df[CATEGORICAL].fillna(-1).astype('int').astype('str')
    return df

@click.command()
@click.option("--year", type=int, required=True, help="Year of the trip data (e.g., 2023)")
@click.option("--month", type=int, required=True, help="Month of the trip data (e.g., 4)")
def main(year, month):
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    input_file = f'../data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(input_file)

    dicts = df[CATEGORICAL].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("Mean predicted duration:", np.mean(y_pred))
    print("Std predicted duration:", np.std(y_pred))

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    taxi_type = 'yellow'
    output_file = f'../output/{taxi_type}-{year:04d}-{month:02d}.parquet'
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

    file_size_bytes = os.path.getsize(output_file)
    print(f"Output file written to: {output_file}")
    print(f"File size: {file_size_bytes / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    main()
