import os

from huggingface_hub.hf_api import HfApi

HEZAR_HUB_ID = 'hezar-ai'
HEZAR_CACHE_DIR = f'{os.path.expanduser("~")}/.hezar'
HEZAR_TMP_DIR = f'{HEZAR_CACHE_DIR}/tmp'
HEZAR_SNAPSHOTS_DIR = f'{HEZAR_CACHE_DIR}/snapshots'
HEZAR_MODELS_CACHE_DIR = f'{HEZAR_CACHE_DIR}/models'
HEZAR_DATASETS_CACHE_DIR = f'{HEZAR_CACHE_DIR}/datasets'
REPO_TYPE_TO_DIR_MAPPING = dict(
    model=HEZAR_MODELS_CACHE_DIR,
    dataset=HEZAR_DATASETS_CACHE_DIR
)


def exists_on_hub(hub_path: str, type='model'):
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
