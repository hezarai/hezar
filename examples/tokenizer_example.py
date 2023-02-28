from hezar.preprocessors.tokenizers import Tokenizer

tokenizer = Tokenizer.load(hub_or_local_path="hezar-ai/distilbert-fa-sentiment-v1")
encoded = tokenizer(["Hello guys", "my name is", "Aryan Shekarlaban"], return_attention_mask=True)
decoded = tokenizer.decode_batch(encoded["token_ids"])
tokenizer.save("misc/wordpiece")
...
