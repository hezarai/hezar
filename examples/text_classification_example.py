from hezar import build_model, Model, Tokenizer

hub_path = "hezar-ai/bert-fa-sentiment-digikala-snappfood"
device = "cpu"
model = Model.load(hub_path).to(device)
tokenizer = Tokenizer.load(hub_path)
inputs = ["کتابخانه هزار، بهترین کتابخانه هوش مصنوعیه"]
model_inputs = tokenizer(inputs, return_tensors="pt", device=device)
model_outputs = model.predict(model_inputs)
print(model_outputs)
