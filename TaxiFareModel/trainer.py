import numpy as np
import time
import category_encoders as ce

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
# from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV

from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, \
    AddGeohash, Direction, DistanceToCenter
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data, DIST_ARGS
from TaxiFareModel.mlf import Mlf

import joblib
from termcolor import colored

class Trainer():
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

        self.kwargs = kwargs
        self.local = kwargs.get("local", False)  # if True training is done locally
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to mlflow
        self.experiment_name = kwargs.get("experiment_name")  # cf doc above
        self.model_params = None


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist = self.kwargs.get("distance_type", "euclidian")
        feateng_steps = self.kwargs.get("feateng", ["distance", "time_features"])

        # Define feature engineering pipeline blocks here
        ## create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer(distance_type=dist, **DIST_ARGS)),
            ('stdscaler', StandardScaler())
        ])


        ## create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        ## create other pipelines
        pipe_geohash = make_pipeline(AddGeohash(), ce.HashingEncoder())
        pipe_direction = make_pipeline(Direction(), StandardScaler())
        pipe_distance_to_center = make_pipeline(DistanceToCenter(), StandardScaler())

        # create preprocessing pipeline/features_encoder
        ## Define default feature engineering blocs
        feateng_blocks = [
            ('distance', dist_pipe, list(DIST_ARGS.values())),
            ('time_features', time_pipe, ['pickup_datetime']),
            # ('geohash', pipe_geohash, list(DIST_ARGS.values())),
            ('direction', pipe_direction, list(DIST_ARGS.values())),
            ('distance_to_center', pipe_distance_to_center, list(DIST_ARGS.values())),
        ]
        ## Filter out some bocks according to input parameters
        for bloc in feateng_blocks:
            if bloc[0] not in feateng_steps:
                feateng_blocks.remove(bloc)

        preproc_pipe = ColumnTransformer(feateng_blocks, n_jobs=None, remainder="drop")

        # Add the model of your choice to the pipeline
        ## LinearRegression
        # self.pipeline = Pipeline([
        #     ('preproc', preproc_pipe),
        #     ('linear_model', LinearRegression())
        # ])

        ## Lasso
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('lasso_model', Lasso()) # found with RandomizedSearch
        ])
        #print(pip.get_params())

    def run(self):
        """set and train the pipeline"""
        # set the pipelined model
        self.set_pipeline()
        # train the pipelined model
        self.pipeline.fit(self.X, self.y)
        return self.pipeline


    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))

# evaluation metric: rmse
    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        # compute y_pred on the test set
        y_pred = self.pipeline.predict(X_test)

        # evaluate the pipeline on df_test
        self.rmse = compute_rmse(y_pred, y_test)
        return round(self.rmse, 2)

# MLFlow Code
    def mlflow_track(self, **kwargs):
      print('kwargs:', kwargs)
      mlf_tracker = Mlf(self.experiment_name)

      for k, v in kwargs.items():
        mlf_tracker.mlflow_log_param(k, v)

      mlf_tracker.mlflow_log_metric('rmse', self.rmse)

      return mlf_tracker.mlflow_experiment_id

# save the model
    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))







if __name__ == "__main__":
    params = dict(local=False,  # set to False to get data from GCP (Storage or BigQuery)
                  optimize=True,
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name='[CN][SH][RoselynYuer] TaxiFareModel v1',
                  distance_type="manhattan",
                  feateng=["distance_to_center", "direction", "distance", "time_features", "geohash"])

    # get data
    N = 1000
    print('started fetching data...')
    df = get_data(nrows=N, **params)

    # clean data
    print('finished downloading data... started cleaning data...')
    df = clean_data(df)

    # set X and y
    print('finished cleaning data... started setting X and y...')
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    print('finished setting X and y... started setting train and test data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # train
    print('finished setting train and test data...started setting and trainining the model...')
    ## initiate the model
    trainer = Trainer(X_train, y_train, **params)
    ## set and train the pipeline
    pipeline = trainer.run()

    # evaluate
    print('finished setting and trainining the model...started evaluating the model...')
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")

    # Randomized Search
    print('---- started randomized search ----')
    ## params can be seen by print(pip.get_params())
    params = {
      "lasso_model__alpha": np.linspace(0.1, 2, 10),
      "lasso_model__max_iter": np.arange(500, 5000, 500, dtype=np.int64),
      "lasso_model__tol": np.linspace(0.0001, 0.1, 10),
      "preproc__distance__stdscaler__with_mean": [True, False]
    }

    rands = RandomizedSearchCV(pipeline, params, n_iter=10, cv=5, n_jobs=-1,\
                               error_score="raise")

    rands.fit(X_train, y_train)

    # Call MLFlow for tracking
    mlflow_experiment_id = trainer.mlflow_track(**rands.best_params_)

    # save the model
    trainer.save_model()

    # print the website to see the result
    print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{mlflow_experiment_id}")
