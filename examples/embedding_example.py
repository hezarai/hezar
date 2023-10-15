from hezar import Embedding


fasttext = Embedding.load("hezarai/word2vec-skipgram-fa-wikipedia")
most_similar = fasttext.most_similar("هزار")
print(most_similar)
