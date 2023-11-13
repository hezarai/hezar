from unittest import TestCase

from hezar.embeddings import Embedding

TESTABLE_EMBEDDINGS = {
    "word2vec-skipgram": "hezarai/word2vec-skipgram-fa-wikipedia",
    "word2vec-cbow": "hezarai/word2vec-cbow-fa-wikipedia",
    "fasttext": "hezarai/fasttext-fa-300",
}
MOST_SIMILAR_TOP_N = 5


class EmbeddingsTestCase(TestCase):
    def test_word2vec_skipgram(self):
        embedding_model = Embedding.load(TESTABLE_EMBEDDINGS["word2vec-skipgram"])
        most_similar = embedding_model.most_similar("هزار", top_n=MOST_SIMILAR_TOP_N)
        self.assertEqual(type(most_similar), list)
        self.assertEqual(len(most_similar), MOST_SIMILAR_TOP_N)
        self.assertEqual(
            {k for x in most_similar for k in x.keys()},
            {"word", "score"}
        )
        doesnt_match = embedding_model.doesnt_match(["کامپیوتر", "مهندس", "نرم افزار", "دوغ"])
        self.assertEqual(type(doesnt_match), str)

    def test_word2vec_cbow(self):
        embedding_model = Embedding.load(TESTABLE_EMBEDDINGS["word2vec-cbow"])
        most_similar = embedding_model.most_similar("هزار", top_n=MOST_SIMILAR_TOP_N)
        self.assertEqual(type(most_similar), list)
        self.assertEqual(len(most_similar), MOST_SIMILAR_TOP_N)
        self.assertEqual(
            {k for x in most_similar for k in x.keys()},
            {"word", "score"}
        )
        doesnt_match = embedding_model.doesnt_match(["کامپیوتر", "مهندس", "نرم افزار", "دوغ"])
        self.assertEqual(type(doesnt_match), str)

    def test_fasttext(self):
        embedding_model = Embedding.load(TESTABLE_EMBEDDINGS["fasttext"])
        most_similar = embedding_model.most_similar("هزار", top_n=MOST_SIMILAR_TOP_N)
        self.assertEqual(type(most_similar), list)
        self.assertEqual(len(most_similar), MOST_SIMILAR_TOP_N)
        self.assertEqual(
            {k for x in most_similar for k in x.keys()},
            {"word", "score"}
        )
        doesnt_match = embedding_model.doesnt_match(["کامپیوتر", "مهندس", "نرم افزار", "دوغ"])
        self.assertEqual(type(doesnt_match), str)
