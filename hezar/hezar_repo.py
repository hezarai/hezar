import os
import shutil
from typing import *
import logging

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf, DictConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.Logger(__name__)
logger.setLevel(20)

HEZAR_HUB_ID = 'hezar-ai'
HEZAR_CACHE_DIR = f'{os.path.expanduser("~")}/.hezar'
HEZAR_SNAPSHOTS_DIR = f'{HEZAR_CACHE_DIR}/snapshots'
HEZAR_MODELS_CACHE_DIR = f'{HEZAR_CACHE_DIR}/models'
HEZAR_DATASETS_CACHE_DIR = f'{HEZAR_CACHE_DIR}/datasets'


class HezarRepo:
    def __init__(self, repo_name):
        self.repo_name = repo_name
        self.repo_id = f'{HEZAR_HUB_ID}/{repo_name}'
        self.repo_dir = self.setup_repo()

    def setup_repo(self):
        repo_dir = self.load_repo()
        return repo_dir

    def load_repo(self, **kwargs):
        target_path = f'{HEZAR_MODELS_CACHE_DIR}/{self.repo_name}'
        repo_dir = snapshot_download(repo_id=self.repo_id, cache_dir=HEZAR_SNAPSHOTS_DIR, **kwargs)
        self.move_repo(repo_dir, target_path, keep_source=True)
        logging.info(f'Initiated repo for `{self.repo_id}` ({target_path})')
        return target_path

    @staticmethod
    def move_repo(source_path: Union[str, os.PathLike], target_path: Union[str, os.PathLike], keep_source=False):
        os.makedirs(target_path, exist_ok=True)
        for file in os.listdir(source_path):
            if file not in os.listdir(target_path):
                shutil.copy2(f'{source_path}/{file}', f'{target_path}/{file}')
        if not keep_source:
            shutil.rmtree(source_path)

    def get_model_registry_name(self):
        config = self.get_config()
        return config.model.name

    def get_config(self, config_file='config.yaml', model_config_class=None):
        config_file_path = os.path.join(self.repo_dir, config_file)
        dict_config = OmegaConf.load(config_file_path)
        if model_config_class:
            config = model_config_class.from_dict(dict_config)
            config.pretrained_path = f'{HEZAR_HUB_ID}/{self.repo_name}'
            return config
        return dict_config

    def get_model(self, model_file='pytorch_model.bin', return_state_dict=True):
        model_file_path = os.path.join(self.repo_dir, model_file)
        if return_state_dict:
            state_dict = torch.load(model_file_path)
            return state_dict
        else:
            return model_file_path
