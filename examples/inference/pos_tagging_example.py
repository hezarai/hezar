from hezar.models import Model


model = Model.load("hezarai/bert-fa-pos-lscp-500k")
inputs = ["سلام بر فارسی زبانان شریف"]
outputs = model.predict(inputs, return_scores=True, return_offsets=True)
print(outputs)
