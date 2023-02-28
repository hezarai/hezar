from hezar import Tokenizer, TokenizerConfig, TokenizerTrainerConfig

data = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
]

train_config = TokenizerTrainerConfig(
    model="wordpiece",
    vocab_size=2000,
    special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
)
t = Tokenizer.train(
    data,
    train_config,
    tokenizer_config=TokenizerConfig(model_kwargs={"unk_token": "[UNK]"}),
)
y = t(["This is just a test!"])
print(y)
