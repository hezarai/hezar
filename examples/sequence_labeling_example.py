from hezar import Model

hub_path = "hezarai/bert-fa-pos-lscp-500k"
model = Model.load(hub_path)
inputs = ["سلام بر فارسی زبانان شریف"]
x = model.predict(inputs)
model.push_to_hub("hezarai/bert-fa-pos-lscp-500k")
print(x)
