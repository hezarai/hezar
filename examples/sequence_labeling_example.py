from hezar import Model

hub_path = "hezarai/bert-fa-pos-lscrp-pos-500k"
model = Model.load(hub_path)
inputs = ["سلام بر فارسی زبانان شریف"]
outputs = model.predict(inputs)
print(outputs)
