from hezar import Tokenizer, TokenizerConfig, TokenizerTrainerConfig

data = [
    "Beautiful is better than ugly."
    "Explicit is better than implicit."
    "Simple is better than complex."
    "Complex is better than complicated."
    "Flat is better than nested."
    "Sparse is better than dense."
    "Readability counts."
    "A quick brown fox jumps over the lazy dog"
    "A QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
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
x = t(["This is a test"], return_tokens=True, return_tensors="pt")
print(x)
y = t.decode(x["token_ids"].numpy().tolist())
print(y)
