def build_models():
    from hezar import list_available_models, build_model

    for name in list_available_models():
        try:
            build_model(name)
        except Exception as e:
            raise e
        print(f"Successfully built `{name}`")


def build_preprocessors():
    from hezar import list_available_preprocessors, build_preprocessor

    for name in list_available_preprocessors():
        try:
            build_preprocessor(name)
        except Exception as e:
            raise e
        print(f"Successfully built `{name}`")


def build_embeddings():
    from hezar import list_available_embeddings, build_embedding

    for name in list_available_embeddings():
        try:
            build_embedding(name)
        except Exception as e:
            raise e
        print(f"Successfully built `{name}`")


def build_datasets():
    from hezar import list_available_datasets, build_dataset

    for name in list_available_datasets():
        try:
            build_dataset(name)
        except Exception as e:
            raise e
        print(f"Successfully built `{name}`")


def build_metrics():
    from hezar import list_available_metrics, build_metric

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
