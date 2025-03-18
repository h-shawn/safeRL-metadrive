import os.path as osp
import torch
import yaml

def load_config_and_model(path: str, best: bool = False):
    """
    Load the configuration and trained model from a specified directory.

    :param path: the directory path where the configuration and trained model are stored.
    :param best: whether to load the best-performing model or the most recent one.
        Defaults to False.

    :return: a tuple containing the configuration dictionary and the trained model.
    :raises ValueError: if the specified directory does not exist.
    """
    if osp.exists(path):
        config_file = osp.join(path, "config.yaml")
        print(f"load config from {config_file}")
        with open(config_file) as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        model_file = "model.pt"
        if best:
            model_file = "model_best.pt"
        model_path = osp.join(path, "checkpoint/" + model_file)
        print(f"load model from {model_path}")
        model = torch.load(model_path, weights_only=False)
        return config, model
    else:
        raise ValueError(f"{path} doesn't exist!")