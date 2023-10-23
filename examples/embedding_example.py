from hezar import Embedding


embedding_model = Embedding.load("hezarai/word2vec-skipgram-fa-wikipedia")
most_similar = embedding_model.most_similar("هزار")
print(most_similar)
