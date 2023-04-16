from hezar import build_model, Model, Tokenizer

hub_path = "hezarai/roberta-fa-sentiment-digikala-snappfood"
model = Model.load(hub_path, device="cpu")
tokenizer = Tokenizer.load(hub_path)
inputs = ["کتابخانه هزار، بهترین کتابخانه هوش مصنوعیه"]
model_inputs = tokenizer(inputs, return_tensors="pt", device="cpu")
model_outputs = model.predict(model_inputs)
print(model_outputs)
