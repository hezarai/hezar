from hezar import Tokenizer, TokenizerTrainerConfig

data = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
]

train_config = TokenizerTrainerConfig(model="wordpiece", vocab_size=20000, special_tokens=["<PAD>", "<BOS>", "<EOS>"])
t = Tokenizer.train(data, train_config)
