from huggingface_hub.hf_api import HfApi


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

    return hub_path in paths
