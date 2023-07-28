from hezar import build_model, Tokenizer

hub_path = "hezarai/bert-base-fa"
tokenizer = Tokenizer.load(hub_path)
# dummy model just to test
model = build_model("bert_sequence_labeling", id2label={0: "NOUN", 1: "VERB", 2: "ADJ", 3: "ADV", 4: "DET"})
model.preprocessor = tokenizer
inputs = ["سلام بر فارسی زبانان شریف"]
x = model.predict(inputs)
print(x)
