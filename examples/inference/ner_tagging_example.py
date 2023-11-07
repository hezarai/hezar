from hezar import Model


model = Model.load("hezarai/bert-fa-pos-lscp-500k")
inputs = ["سلام بر فارسی زبانان شریف"]
outputs = model.predict(inputs)
print(outputs)
