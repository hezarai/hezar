from hezar import Model

model = Model.load("hezarai/bert-fa-ner-arman")
inputs = ["شرکت هوش مصنوعی هزار"]
outputs = model.predict(inputs)
print(outputs)
