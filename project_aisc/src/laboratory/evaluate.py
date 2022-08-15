import mlflow
import numpy as np
import pathlib
import tensorflow as tf
import json

def main():
    # Path to get inside project_aisc folder
    home_path = str(pathlib.Path(__file__).absolute().parent.parent.parent)
    with open(home_path+"/../active_experiments.json") as experiments_json:
        experiment_info = json.load(experiments_json)

    run_id = experiment_info["run id"]
    experiment_id = experiment_info["experiment id"]
    model_uri = home_path+"/data/experiments/mlruns/"+experiment_id+"/"+run_id+"/artifacts/model"
    model = mlflow.keras.load_model(model_uri)

    X_test = list(np.load(home_path + '/data/processed/test/X_test.npy'))
    Y_test = np.load(home_path + '/data/processed/test/Y_test.npy')

    Y_pred = tf.reshape(model.predict(X_test),shape=(-1,))
    mae = tf.keras.metrics.mean_absolute_error(Y_pred,Y_test)
    mse = tf.keras.metrics.mean_squared_error(Y_pred, Y_test)
    print(f"MAE : {mae}\nMSE : {mse}\nRMSE : {tf.math.sqrt(mse)}")

    MlflowClient = mlflow.tracking.client.MlflowClient(tracking_uri = home_path + "/data/experiments/mlruns/")

    MlflowClient.log_metric(run_id=run_id, key='test_root_mean_squared_error', value=float(mae))
    MlflowClient.log_metric(run_id=run_id, key='test_mean_squared_error', value=float(mse))
    MlflowClient.log_metric(run_id=run_id, key='test_mean_absolute_error', value=float(tf.math.sqrt(mse)))

    artifact_uri = home_path+"/data/experiments/mlruns/"+experiment_id+"/"+run_id+"/artifacts
    # Save predictions of the model
    np.save(artifact_uri + "/predictions.npy", tf.reshape(model.predict(X_test), shape=(-1,)).numpy())


if __name__ == '__main__':
    main()
