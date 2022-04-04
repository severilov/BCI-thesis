from models.lnn import init_lnn
from models.lstm import init_lstm


def init_model(model_type: str,
               params: dict):
    """
    :param model_type:
    :param params:
    :return:
    """
    if model_type == "lnn":
        return init_lnn(**params)
    elif model_type == "lstm":
        return init_lstm(**params)
    raise NotImplementedError