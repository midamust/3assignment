import pandas as pd
import mlflow

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.

# mlflow.set_tracking_uri("http://training.itu.dk:5000/")

# TODO: Set the experiment name
mlflow.set_experiment("<mids> - <MLFlow_experiemnts>")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
import math

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


class Debug(BaseEstimator, TransformerMixin):
    def fit(self, X, y): return self

    def transform(self, X):
        print(X)
        return X

# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="<TestRun>"):
    # TODO: Insert path to dataset
    df = pd.read_json("dataset.json", orient="split")

    # TODO: Handle missing data
    df = df.dropna()

    X = df[["Direction", "Lead_hours", "Source_time", "Speed"]]
    y = df[["Total"]]
    
    pipeline = Pipeline([
        # TODO: You can start with your pipeline from assignment 1
        ("Direction to Radians", Direction_To_Radians()),
        ("Drop ", ColumnTransformer([
            ("Speed", "passthrough", ["Speed"]),
            ("DirectionRad", "passthrough", ["DirectionRad"]),
        ], remainder="drop")),
        #("Debug", Debug()),
        ("Scaler", MinMaxScaler()),
        ("LinearRegressionModel", LinearRegression())
    ])

    # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
    metrics = [
        ("MAE", mean_absolute_error, []),
        ("MSE", mean_squared_error, []),
        ("R2", r2_score, []),
    ]

    X = df[["Speed","Direction"]]
    y = df["Total"]

    number_of_splits = 5

    #TODO: Log your parameters. What parameters are important to log?
    #HINT: You can get access to the transformers in your pipeline using `pipeline.steps`
    
    for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]
        
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics:
            score = func(truth, predictions)
            scores.append(score)
    
    # Log a summary of the metrics
    for name, _, scores in metrics:
            # NOTE: Here we just log the mean of the scores. 
            # Are there other summarizations that could be interesting?
            mean_score = sum(scores)/number_of_splits
            mlflow.log_metric(f"mean_{name}", mean_score)
