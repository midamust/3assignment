import mlflow
import numpy as np
import pandas as pd
from numpy.lib.function_base import median, select
from matplotlib import pyplot as plt

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.

# mlflow.set_tracking_uri("http://training.itu.dk:5000/")

# TODO: Set the experiment name
mlflow.set_experiment("mids - Analytics: (Sampling, Model:hyperParam, Transformer, Split) ")

import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler, OneHotEncoder,
                                   PolynomialFeatures, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVR


def met_degrees_to_math(met_dir):
    degrees_math = (270 - met_dir) % 360
    return degrees_math


class Direction_To_Radians(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        met_dir_degrees = {
            "W": 270.0,
            "WSW": 247.5,
            "SSW": 202.5,
            "SW": 225.0,
            "S": 180.0,
            "SE": 135,
            "SSE": 157.5,
            "ESE": 112.5,
            "E": 90.0,
            "NE": 45.0,
            "NNE": 22.5,
            "ENE": 67.5,
            "N": 0.0,
            "NW": 315.0,
            "NNW": 337.5,
            "WNW": 292.5,
        }
        X["DirectionRad"] = [math.radians(met_degrees_to_math(
            met_dir_degrees.get(i))) for i in X["Direction"].values]
        return X

class ToUV(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        text_to_num = { "N": 0, "NNE": 1, "NE": 2, "ENE": 3, "E": 4, "ESE": 5, "SE": 6, "SSE": 7, "S": 8, "SSW": 9, "SW": 10, "WSW": 11, "W": 12, "WNW": 13, "NW": 14, "NNW": 15 }
        dirs = X["Direction"].replace(text_to_num)
        radians =  np.radians((270 - dirs * 22.5) % 360)
        us = X["Speed"] * np.cos(radians)
        vs = X["Speed"] * np.sin(radians)
        uvs = pd.concat([us, vs], axis=1, sort=False)
        return uvs

class Debug(BaseEstimator, TransformerMixin):
    def fit(self, X, y): return self

    def transform(self, X):
        print(X)
        return X

class Dropna(BaseEstimator, TransformerMixin):
    def fit(self, X, y): return self

    def transform(self, X):
        X = X.dropna()
        return X

# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="Radians, Linreg"):
    # TODO: Insert path to dataset
    df = pd.read_json("dataset.json", orient="split")

    # TODO: Handle missing data
    df = df.dropna()

    #Specify hyperparameter for polynomial degree
    degree = 1

    pipeline = Pipeline([
        # TODO: You can start with your pipeline from assignment 1
        ("DropNA", Dropna()),
        ("Direction to Radians", Direction_To_Radians()),
        ("Drop ", ColumnTransformer([
            ("Speed", "passthrough", ["Speed"]),
            #("OneHotEncoder", OneHotEncoder(), ["Direction"]),
            ("DirectionRad", "passthrough", ["DirectionRad"]),
        ], remainder="drop")),
        #("Debug", Debug()),
        ("Scaler", MaxAbsScaler()),
        #("Poly", PolynomialFeatures(degree=degree)),
        ("LinearRegressionModel", LinearRegression())
        #("SVRRegressionModel", SVR())
        
    ])

    metrics = [
    ("MAE", mean_absolute_error, []),
    ("MSE", mean_squared_error, []),
    ("R2", r2_score, []),
    ]

    X = df[["Speed","Direction"]]
    y = df["Total"]


    splits = 5
    #TODO: Log your parameters. What parameters are important to log?
    #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
    mlflow.log_param("Model", pipeline.steps[-1])
    mlflow.log_param("Pipeline setup", pipeline.steps)
    mlflow.log_param("Splits", splits)
    mlflow.log_param("polyDegree", degree)

    # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?

    # current split
    split = 0
    for train, test in TimeSeriesSplit(splits).split(X,y):
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]
        
        split = split + 1

        plt.plot(X.iloc[test].Speed, truth, "--", label="Truth")
        plt.plot(X.iloc[test].Speed, predictions, label="Predictions")
        plt.xlabel("WindSpeed")
        plt.ylabel("Power Generation in MegaWatts")
        plt.legend()
        plt.savefig(str(split) + "split.png")
        mlflow.log_artifact(str(split) + "split.png")
        plt.close()

        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)

    # Log a summary of the metrics
    for name, _, scores in metrics:
            # NOTE: Here we just log the mean of the scores. 
            # Are there other summarizations that could be interesting?
            mean_score = sum(scores)/splits
            mlflow.log_metric(f"mean_{name}", mean_score)
            n = len(scores)
            scores.sort()
            median_score = scores[n // 2]
            mlflow.log_metric(f"median_{name}", median_score)

