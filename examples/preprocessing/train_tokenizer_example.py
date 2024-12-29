from hezar.preprocessors import (
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

def train_and_test_tokenizer(tokenizer_class, config_class, data):
    tokenizer_config = config_class()
    tokenizer = tokenizer_class(tokenizer_config)
    tokenizer.train_from_iterator(dataset=data)
    x = tokenizer(["This is a test"], return_tokens=True, return_tensors="torch")
    print(x)
    y = tokenizer.decode(x["token_ids"].numpy().tolist())
    print(y)

train_and_test_tokenizer(WordPieceTokenizer, WordPieceConfig, data)
train_and_test_tokenizer(BPETokenizer, BPEConfig, data)
train_and_test_tokenizer(SentencePieceBPETokenizer, SentencePieceBPEConfig, data)
train_and_test_tokenizer(SentencePieceUnigramTokenizer, SentencePieceUnigramConfig, data)
