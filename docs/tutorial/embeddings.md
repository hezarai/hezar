# Embeddings
In Hezar, embeddings serve as fundamental components for various natural language processing tasks. The Embedding class
provides a flexible and extensible foundation for working with word embeddings. Currently Hezar has two embedding models
backed by Gensim. This tutorial will guide you through the essential aspects of using and customizing embeddings in Hezar.

## Load an Embedding from Hub
Loading an embedding from a pretrained embedding on the Hub or locally, is as straightforward as other modules in Hezar.
You can choose your desired model from our Hub and load it like below:
```python
from hezar.embeddings import Embedding

word2vec = Embedding.load("hezarai/word2vec-cbow-fa-wikipedia")
```
Now let's just run a simple similarity test between two given words:
```python
word2vec.similarity("هزار", "میلیون")
```
```
0.7400991
```

## Embeddings methods

### Similarity
For getting the similarity score between two words, use the following:
```python
similarity_score = word2vec.similarity("سلام", "درود")
print(similarity_score)
```
```
0.6196184
```
### Get Top-n Similar Words
Find top-n most similar words to a given word like:
```python
from pprint import pprint

most_similar = word2vec.most_similar("هزار", topn=5)
pprint(most_similar)
```
```
[{'score': '0.7407', 'word': 'دویست'},
 {'score': '0.7401', 'word': 'میلیون'},
 {'score': '0.7326', 'word': 'صد'},
 {'score': '0.7277', 'word': 'پانصد'},
 {'score': '0.7011', 'word': 'سیصد'}]
```

### Least Similar in a List
To get the least similar word in a list or a word that does not match other words in a list, use the following:
```python
least_similar = word2vec.doesnt_match(["خانه", "اتاق", "ماشین"])
```
```
'ماشین'
```

### Get Word's Vector
Get the vector for a word by:
```python
vector = word2vec("سلام")
```
You can also give the model a list of words to get vectors for each of them:
```python
vectors = word2vec(["هوش", "مصنوعی"])
```

### Get the Vocabulary
Get the dictionary of the whole vocabulary in the embedding model:
```python
vocab = word2vec.vocab
```
#### Vocabulary words and indexes
You can also get index of a word in the vocabulary or vise verse:
```python
index = word2vec.word2index("هوش")
word = word2vec.index2word(index)
print(word)
```
```
'هوش'
```
### Converting to a PyTorch nn.Embedding
You can also get a PyTorch embedding layer from the embedding model:
```python
embedding_layer = word2vec.torch_embedding()
print(embedding_layer)
```
```
Embedding(240547, 200)
```
## Training an Embedding Model
To train an embedding model, first choose and build your embedding. For this example, we'll train a Word2Vec model using
the CBOW algorithm with a vector dimension of 200.
```python
from hezar.embeddings import Word2Vec, Word2VecConfig

model = Word2Vec(
    Word2VecConfig(
        vector_size=200,
        window=5,
        train_algorithm="cbow",
        alpha=0.025,
        min_count=1,
        seed=1,
        workers=4,
        min_alpha=0.0001,
    )
)
```
Now given a list of sentences as the dataset, run training process:
```python
with open("data.txt") as f:
    sentences = f.readlines()

sentences = [s.replace("\n", "") for s in sentences]

word2vec.train(sentences, epochs=5)
```
## Saving and Pushing to the Hub 
Now you can save and push your model to the Hub:
```python
word2vec.save("word2vec-cbow-200")

word2vec.push_to_hub("<your-hf-username>/word2vec-cbow-200-fa")
```


