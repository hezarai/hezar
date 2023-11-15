from hezar.embeddings import Embedding
import pytest

TESTABLE_EMBEDDINGS = {
    "word2vec-skipgram": "hezarai/word2vec-skipgram-fa-wikipedia",
    "word2vec-cbow": "hezarai/word2vec-cbow-fa-wikipedia",
    "fasttext": "hezarai/fasttext-fa-300",
}
MOST_SIMILAR_TOP_N = 5


@pytest.mark.parametrize("embedding_key", TESTABLE_EMBEDDINGS.keys())
def test_embedding_most_similar(embedding_key):
    embedding_model = Embedding.load(TESTABLE_EMBEDDINGS[embedding_key])
    most_similar = embedding_model.most_similar("هزار", top_n=MOST_SIMILAR_TOP_N)
    assert isinstance(most_similar, list)
    assert len(most_similar) == MOST_SIMILAR_TOP_N
    assert {k for x in most_similar for k in x.keys()} == {"word", "score"}


@pytest.mark.parametrize("embedding_key", TESTABLE_EMBEDDINGS.keys())
def test_embedding_doesnt_match(embedding_key):
    embedding_model = Embedding.load(TESTABLE_EMBEDDINGS[embedding_key])
    doesnt_match = embedding_model.doesnt_match(["کامپیوتر", "مهندس", "نرم افزار", "دوغ"])
    assert isinstance(doesnt_match, str)
