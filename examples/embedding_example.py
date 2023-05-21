from hezar import build_embedding

e = build_embedding("word2vec_cbow")

e.push_to_hub("arxyzan/embedding_test", private=True)
...