from hezar import Embedding

fasttext = Embedding.load("hezarai/fasttext-fa-300")
most_similar = fasttext.most_similar("هزار")
print(most_similar)
