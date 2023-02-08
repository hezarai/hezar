import os
from typing import *

import torch
from huggingface_hub import Repository, HfApi, create_repo
from omegaconf import OmegaConf, DictConfig

from hezar.utils.logging import get_logger

HEZAR_HUB_ID = 'hezar-ai'
HEZAR_CACHE_DIR = os.getenv('HEZAR_CACHE_DIR', f'{os.path.expanduser("~")}/.hezar')
HEZAR_TMP_DIR = os.getenv('HEZAR_TMP_DIR', f'{os.path.expanduser("~")}/.cache/hezar')
HEZAR_SNAPSHOTS_DIR = os.getenv('HEZAR_SNAPSHOTS_DIR', f'{HEZAR_CACHE_DIR}/snapshots')
HEZAR_MODELS_CACHE_DIR = os.getenv('HEZAR_MODELS_CACHE_DIR', f'{HEZAR_CACHE_DIR}/models')
HEZAR_DATASETS_CACHE_DIR = os.getenv('HEZAR_DATASETS_CACHE_DIR', f'{HEZAR_CACHE_DIR}/datasets')
REPO_TYPE_TO_DIR_MAPPING = dict(
    model=HEZAR_MODELS_CACHE_DIR,
    dataset=HEZAR_DATASETS_CACHE_DIR
)

logger = get_logger(__name__)


def resolve_hub_path(hub_path):
    """
    If hub_path contains the namespace (author/org) leave it as is, otherwise change to hezar-ai/{hub_path}

    Args:
        hub_path: repo name or id

    Returns:
        A proper repo id on the hub
    """
    if os.path.isdir(hub_path):
        logger.warning(f'{hub_path} does not seem to be a valid or existing repo on the Hub!')
    repo_id = f'{HEZAR_HUB_ID}/{hub_path}' if '/' not in hub_path else hub_path
    return repo_id


def get_local_cache_path(hub_path, repo_type):
    """
    Given the hub path and repo type, configure the local path to save everything e.g, ~/.hezar/models/<repo_name>

    Args:
        hub_path: repo name or id
        repo_type: repo type e.g, model, dataset, etc

    Returns:
        path to local cache directory
    """
    repo_id = resolve_hub_path(hub_path)
    repo_name = repo_id.split('/')[1]
    cache_path = f'{REPO_TYPE_TO_DIR_MAPPING[repo_type]}/{repo_name}'
    return cache_path


def exists_on_hub(hub_path: str, type='model'):
    """
    Determine whether the repo exists on the hub or not

    Args:
        hub_path: repo name or id
        type: repo type like model, dataset, etc.

    Returns:
        True or False
    """
    author, repo_name = hub_path.split('/')
    api = HfApi()
    if type == 'model':
        paths = list(iter(api.list_models(author=author)))
    elif type == 'dataset':
        paths = list(iter(api.list_datasets(author=author)))
    elif type == 'space':
        paths = list(iter(api.list_spaces(author=author)))
    else:
        raise ValueError(f'Unknown type: {type}! Use `model`, `dataset`, `space`, etc.')

    return hub_path in [path.id for path in paths]


def clone_repo(hub_path: str, save_path: str, **kwargs):
    """
    Clone a repo on the hub to local directory

    Args:
        hub_path: repo name or id
        save_path: path to clone the repo to

    Returns:
        the local path to the repo
    """
    repo_id = resolve_hub_path(hub_path)
    repo = Repository(local_dir=save_path, clone_from=repo_id, **kwargs)
    return repo.local_dir
