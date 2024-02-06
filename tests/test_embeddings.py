import os

import pytest

from hezar.embeddings import Embedding
from hezar.utils import clean_cache

CI_MODE = os.environ.get("CI_MODE", "FALSE")


TESTABLE_EMBEDDINGS = {
    "word2vec-skipgram": "hezarai/word2vec-skipgram-fa-wikipedia",
    "word2vec-cbow": "hezarai/word2vec-cbow-fa-wikipedia",
    "fasttext": "hezarai/fasttext-fa-300",
}
MOST_SIMILAR_TOP_N = 5


@pytest.mark.parametrize("embedding_key", TESTABLE_EMBEDDINGS.keys())
def test_embeddings(embedding_key):
    embedding_model = Embedding.load(TESTABLE_EMBEDDINGS[embedding_key])
    most_similar = embedding_model.most_similar("هزار", top_n=MOST_SIMILAR_TOP_N)
    assert isinstance(most_similar, list)
    assert len(most_similar) == MOST_SIMILAR_TOP_N
    assert {k for x in most_similar for k in x.keys()} == {"word", "score"}
    doesnt_match = embedding_model.doesnt_match(["کامپیوتر", "مهندس", "نرم افزار", "دوغ"])
    assert isinstance(doesnt_match, str)

    if CI_MODE == "TRUE":
        clean_cache(delay=1)
