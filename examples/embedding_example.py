from hezar import build_embedding
from hezar import Embedding

emb = Embedding.load("arxyzan/embedding_test")
word2vec = build_embedding("word2vec")
corpus = "This is our test corpus to see if this works".split(" ")
word2vec.train(corpus)
word2vec.push_to_hub("arxyzan/embedding_test", private=True)
