from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import multiprocessing
from psutil import virtual_memory
import mlflow

class Mlf():

  def __init__(self, experiment_name):
    self.experiment_name = experiment_name

  @memoized_property
  def mlflow_client(self):
    MLFLOW_URI = "https://mlflow.lewagon.ai/"
    mlflow.set_tracking_uri(MLFLOW_URI) # set for a remote MLFlow server
    return MlflowClient()

  @memoized_property
  def mlflow_experiment_id(self):
    try:
        return self.mlflow_client.create_experiment(self.experiment_name)
    except BaseException:
        return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

  @memoized_property
  def mlflow_run(self):
    return self.mlflow_client.create_run(self.mlflow_experiment_id)

  # use two functions below to train the model(pipeline): key is param/metric name, value is the param/metric trained
  def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

  def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

  def log_estimator_params(self):
      reg = self.get_estimator()
      self.mlflow_log_param('estimator_name', reg.__class__.__name__)
      params = reg.get_params()
      for k, v in params.items():
          self.mlflow_log_param(k, v)

  def log_machine_specs(self):
      cpus = multiprocessing.cpu_count()
      mem = virtual_memory()
      ram = int(mem.total / 1000000000)
      self.mlflow_log_param("ram", ram)
      self.mlflow_log_param("cpus", cpus)
