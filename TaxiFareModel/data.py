import pandas as pd
import os

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    # get train data as X,y because we get data from train.csv
    #df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    csv_path = os.path.join(os.path.dirname(__file__), '../raw_data/train.csv')
    df = pd.read_csv(csv_path, nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


# if __name__ == '__main__':
#     df = get_data()
#     df_c=clean_data(df)
#     print(df_c.head())


if __name__ == '__main__':
    df = get_data()
