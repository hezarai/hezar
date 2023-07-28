from hezar import build_model, Model, Tokenizer


path = "hezarai/roberta-base-fa"
tokenizer = Tokenizer.load(path)

text = ["Hello Hezar!"]

model = build_model("roberta_lm")
model.preprocessor = tokenizer
outputs = model.predict(text, output_attentions=True, output_hidden_states=True)
print(outputs)

model = Model.load(path)
outputs = model.predict(text, output_attentions=True, output_hidden_states=True)
print(outputs)
