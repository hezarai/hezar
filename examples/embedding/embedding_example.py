from pprint import pprint

from hezar.embeddings import Embedding

embedding_model = Embedding.load("hezarai/fasttext-fa-300")
most_similar = embedding_model.most_similar("هزار")
doesnt_match = embedding_model.doesnt_match(["کامپیوتر", "مهندس", "نرم افزار", "دوغ"])
pprint(most_similar)
