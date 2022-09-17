from hezar.hezar_repo import HezarRepo
from hezar.models import models_registry


def test_repo():
    repo_id = 'test'
    repo = HezarRepo(repo_id)
    config = repo.get_config()
    print(config)


if __name__ == '__main__':
    # test_hub_utils()
    test_repo()
