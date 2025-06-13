import os
import sys
import pickle
import click
import mlflow
from scipy import sparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    # Load data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    X_train = sparse.csr_matrix(X_train)
    X_val = sparse.csr_matrix(X_val)

    # Train model
    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    # Log model size in bytes
    model_bytes = pickle.dumps(rf)
    model_size = sys.getsizeof(model_bytes)
    mlflow.log_metric("model_size_bytes", model_size)

if __name__ == '__main__':
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        run_train()
