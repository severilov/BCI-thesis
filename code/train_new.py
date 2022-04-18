import os
import torch

from tqdm import tqdm

from utils.logger import Logger, log_metrics, setup_mlflow
from utils.tools import clean_device, get_device, get_biases_params, init_optimizer

#from evaluate import evaluate # TODO
from models import init_model
#from datasets import get_dataset, get_dataloader # TODO


def train(model,
          loader,
          criterion,
          optimizer,
          device,
          num_iter,
          epoch,
          cfg):
    pass

def main(cfg: dict):
    """
    :param cfg:
    :return:
    """
    device = get_device(cfg)

    model = init_model(cfg["model"]["type"], cfg["model"]["params"])

    criterion = ':(' # TODO

    biases, not_biases = get_biases_params(model)

    optimizer = init_optimizer(cfg["train"]["optimizer"],
                               biases,
                               not_biases)


    train_datasets = get_dataset(cfg, "train")
    train_loader = get_dataloader(train_datasets, cfg)

    eval_datasets = get_dataset(cfg, "eval")
    eval_loader = get_dataloader(eval_datasets, cfg)

    model = model.to(device)

    num_iter = 0
    best_mse = cfg["train"]["min_mse"]

    for epoch in range(1, cfg["train"]["epoch"] + 1):

        num_iter = train(model,
                         train_loader,
                         criterion,
                         optimizer,
                         device,
                         num_iter,
                         epoch,
                         cfg)

        clean_device(device)

        metrics = evaluate() # TODO

        clean_device(device)

        path2best_model = None

        if metrics["mse"] > best_mse:
            best_mse = metrics["mse"]
            torch.save(model.state_dict(), os.path.join(weights_folder, f"{run_name}.pth"))
            path2best_model = os.path.join(weights_folder, f"{run_name}.pth")

        log_metrics(metrics, epoch, "val", logger, config["log_mlflow"], path2best_model=path2best_model)


if __name__ == '__main__':
    from datetime import datetime
    from utils.tools import get_config

    run_name = datetime.now().strftime("%d_%m_%Y.%H_%M")
    config = get_config()

    log_mlflow = config.get("mlflow", None) is not None
    config["log_mlflow"] = log_mlflow
    config["run_name"] = run_name
    setup_mlflow(config)

    weights_folder = os.path.join(os.getcwd(), "weights",
                                  config["description"]["model_name"],
                                  config["description"]["task_type"])
    log_folder = os.path.join(os.getcwd(),
                              "logs",
                              config["description"]["model_name"],
                              config["description"]["task_type"],
                              run_name)

    os.makedirs(weights_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    logger = Logger(log_folder)

    main(config)
