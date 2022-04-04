import os
import mlflow

from tensorboardX import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, metric_name, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(metric_name, value, step)
        self.writer.flush()


def setup_mlflow(config: dict) -> None:
    """
    """
    if not config["log_mlflow"]:
        return
    # Если передан remote uri - соединяемся, иначе логируем все локально
    if config["mlflow"].get("uri", None):
        print("Connecting to MLflow server ...")
        os.environ['MLFLOW_TRACKING_USERNAME'] = config["mlflow"]["username"]
        os.environ['MLFLOW_TRACKING_PASSWORD'] = config["mlflow"]["password"]
        mlflow.set_tracking_uri(config["mlflow"]["uri"])
    else:
        print("Local running MLflow ...")
        local_path_mlflow = os.path.normpath(config["mlflow"]["localhost"])
        os.makedirs(local_path_mlflow, exist_ok=True)
        mlflow.set_tracking_uri(rf"file://{local_path_mlflow}")

    mlflow.set_experiment(f'{config["description"]["model_name"]}/{config["description"]["task_type"]}')
    mlflow.start_run(run_name=f'{config["description"]["model_name"]}', tags={})
    mlflow.log_artifact(config["config_path"])

    print("Successfull connection!")


def log_metrics(metrics: dict,
                epoch: int = None,
                mode: str = "val",
                logger: Logger = None,
                log_mlflow: bool = False,
                print_result: bool = True,
                path2best_model: str = None) -> None:
    """
    """

    if logger is not None:
        logger.scalar_summary("MSE", metrics, epoch)

    if print_result:
        print("Metrics")

    if log_mlflow:
        mlflow.log_metric('MSE', metrics['mse'], step=epoch)

    if path2best_model is not None:
        print("Saving best model to mlflow ...")
        mlflow.log_artifact(path2best_model)