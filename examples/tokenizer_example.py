from hezar.registry import build_preprocessor
from hezar.preprocessors.tokenizer import Tokenizer


preprocessor = build_preprocessor("tokenizer", pretrained_path="hezar-ai/distilbert-fa")
preprocessor2 = Tokenizer.load(hub_or_local_path="hezar-ai/distilbert-fa")
print(preprocessor(["hello guys"]))
print(preprocessor2((["hello guys"])))
