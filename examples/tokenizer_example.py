from hezar import Tokenizer


tokenizer = Tokenizer.load(hub_or_local_path="hezarai/roberta-base-fa")
encoded = tokenizer(["Hello guys", "my name is", "Aryan Shekarlaban"], return_attention_mask=True)
decoded = tokenizer.decode(encoded["token_ids"])
print(encoded)
print(decoded)
tokenizer = Tokenizer.load(hub_or_local_path="hezarai/distilbert-fa")
encoded = tokenizer(["Hello guys", "my name is", "Aryan Shekarlaban"], return_attention_mask=True)
decoded = tokenizer.decode(encoded["token_ids"])
print(encoded)
print(decoded)
