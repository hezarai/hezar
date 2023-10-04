from hezar import Embedding, Logger


embedding_to_example_repo = {
    "word2vec-cbow": "hezarai/word2vec-cbow-fa-wikipedia",
    "word2vec-skipgram": "hezarai/word2vec-skipgram-fa-wikipedia",
    "fasttext": "hezarai/fasttext-fa-300"
}

logger = Logger(__name__)


def test_embedding_loading():
    embeddings = []
    for e, repo in embedding_to_example_repo.items():
        embedding = Embedding.load(repo)
        embeddings.append(embedding)
        logger.info(f"Successfully loaded {repo} for {e}")
    return embeddings


def test_embedding_saving():
    embeddings = test_embedding_loading()
    for e in embeddings:
        e.save(e.config.name)
        logger.info(f"Successfully saved {e.config.name}")
