
import pickle
import pandas as pd
import numpy as np
import argparse


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')    
    
    return df

def prepare_results(df, y_pred, output_file):

    df_result = pd.DataFrame()

    df_result['ride_id'] = df['ride_id']
    df_result['predictions'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    return df_result

def run():

    parser = argparse.ArgumentParser(description="Receiving year/month inputs")
    parser.add_argument('year', type=int, help="input year")
    parser.add_argument('month', type=int, help="input month")
    args=parser.parse_args()

    year = args.year
    month = args.month
    output_file = "./output/yellow_taxi_predictions.parquet"

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df_results = prepare_results(df, y_pred, output_file)

    print(np.mean(df_results['predictions']))

if __name__ == "__main__":
    run()



