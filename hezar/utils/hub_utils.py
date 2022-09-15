import os

import torch
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

HEZAR_REPO_ID = 'hezar-ai'
HEZAR_CACHE_DIR = f'{os.path.expanduser("~")}/.hezar'


def load_config_from_hub(path_name: str, filename='config.yaml'):
    """
    Load config.yaml from HuggingFace Hub
    """
    repo_id = f'{HEZAR_REPO_ID}/{path_name}'
    cached_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    dict_config = OmegaConf.load(cached_file_path)

    return dict_config


def load_state_dict_from_hub(path_name: str, filename='pytorch_model.bin'):
    """
    Load a PyTorch model state dict from HuggingFace Hub
    """
    repo_id = f'{HEZAR_REPO_ID}/{path_name}'
    cached_file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model_state_dict = torch.load(cached_file_path)

    return model_state_dict
