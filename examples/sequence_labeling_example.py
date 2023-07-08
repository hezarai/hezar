from hezar import build_model, Tokenizer

hub_path = "hezarai/bert-base-fa"
tokenizer = Tokenizer.load(hub_path)
# dummy model just to test
model = build_model("bert_sequence_labeling", id2label={0: "NOUN", 1: "VERB", 2: "ADJ", 3: "ADV", 4: "DET"})
inputs = ["سلام بر فارسی زبانان شریف"]
model_inputs = tokenizer(inputs, return_word_ids=True, return_tokens=True, padding=True, truncation=True,
                         return_tensors="pt")
x = model.predict(model_inputs)
print(x)
