from hezar import build_embedding
from hezar import Embedding

word2vec = build_embedding("word2vec")
corpus = "This is our test corpus to see if this works".split(" ")
corpus = [[x] for x in corpus]
word2vec.train(corpus)
word2vec.push_to_hub("arxyzan/embedding_test", private=True)
