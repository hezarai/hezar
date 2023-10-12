from hezar.preprocessors.tokenizers import (
    BPEConfig,
    BPETokenizer,
    SentencePieceBPEConfig,
    SentencePieceBPETokenizer,
    SentencePieceUnigramConfig,
    SentencePieceUnigramTokenizer,
    WordPieceConfig,
    WordPieceTokenizer,
)

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

tokenizer_config = WordPieceConfig()
tokenizer = WordPieceTokenizer(tokenizer_config)
tokenizer.train_from_iterator(dataset=data)
x = tokenizer(["This is a test"], return_tokens=True, return_tensors="pt")
print(x)
y = tokenizer.decode(x["token_ids"].numpy().tolist())
print(y)

tokenizer_config = BPEConfig()
tokenizer = BPETokenizer(tokenizer_config)
tokenizer.train_from_iterator(dataset=data)
x = tokenizer(["This is a test"], return_tokens=True, return_tensors="pt")
print(x)
y = tokenizer.decode(x["token_ids"].numpy().tolist())
print(y)

tokenizer_config = SentencePieceBPEConfig()
tokenizer = SentencePieceBPETokenizer(tokenizer_config)
tokenizer.train_from_iterator(dataset=data)
x = tokenizer(["This is a test"], return_tokens=True, return_tensors="pt")
print(x)
y = tokenizer.decode(x["token_ids"].numpy().tolist())
print(y)

tokenizer_config = SentencePieceUnigramConfig()
tokenizer = SentencePieceUnigramTokenizer(tokenizer_config)
tokenizer.train_from_iterator(dataset=data)
x = tokenizer(["This is a test"], return_tokens=True, return_tensors="pt")
print(x)
y = tokenizer.decode(x["token_ids"].numpy().tolist())
print(y)
