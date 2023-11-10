def build_models():
    from hezar.models import build_model, list_available_models

    for name in list_available_models():
        try:
            build_model(name)
        except Exception as e:
            raise e
        print(f"Successfully built `{name}`")


def build_preprocessors():
    from hezar.builders import build_preprocessor
    from hezar.utils import list_available_preprocessors

    for name in list_available_preprocessors():
        try:
            build_preprocessor(name)
        except Exception as e:
            raise e
        print(f"Successfully built `{name}`")


def build_embeddings():
    from hezar.builders import build_embedding
    from hezar.utils import list_available_embeddings

    for name in list_available_embeddings():
        try:
            build_embedding(name)
        except Exception as e:
            raise e
        print(f"Successfully built `{name}`")


def build_datasets():
    from hezar.builders import build_dataset
    from hezar.utils import list_available_datasets

    for name in list_available_datasets():
        try:
            build_dataset(name)
        except Exception as e:
            raise e
        print(f"Successfully built `{name}`")


def build_metrics():
    from hezar.builders import build_metric
    from hezar.utils import list_available_metrics

    for name in list_available_metrics():
        try:
            build_metric(name)
        except Exception as e:
            raise e
        print(f"Successfully built `{name}`")


if __name__ == "__main__":
    build_models()
    build_preprocessors()
    build_metrics()
    build_datasets()
    build_embeddings()
