from hezar.builders import build_preprocessor
from hezar.preprocessors.tokenizer import Tokenizer


preprocessor = Tokenizer.load(hub_or_local_path="hezar-ai/distilbert-fa-sentiment-v1")
y1 = preprocessor(["hello guys", "my name is", "Aryan"], return_attention_mask=True, return_tensors="pt")
...

