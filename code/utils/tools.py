import yaml
import torch
import argparse

from typing import List, Tuple


def init_optimizer(cfg,
                   biases,
                   not_biases):
    """
    :param cfg:
    :param biases:
    :param not_biases:
    :return:
    """
    optimizer = eval(f"optim.{cfg['type']}")(
        params=[{'params': biases, 'lr': 2 * cfg["params"]["lr"]}, {'params': not_biases}],
        **cfg['params'])
    return optimizer

def get_config() -> dict:
    parser = argparse.ArgumentParser(description='Read config')
    parser.add_argument("-c", "--config",
                        required=True,
                        help="path to config")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.full_load(f)

    config['config_path'] = args.config

    return config


def clean_device(device: torch.device) -> None:
    """
    :param device:
    :return:
    """
    if device.type != "cpu":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def get_device(cfg: dict) -> torch.device:
    """
    :param cfg:
    :return:
    """

    gpu_index = cfg['train']['gpu_index']

    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

    try:
        torch.cuda.set_device(device)
    except ValueError:
        print(f"Cuda device {device} not found")

    return device

def get_biases_params(model: torch.nn.Module) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    :param model:
    :return:
    """
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    return biases, not_biases