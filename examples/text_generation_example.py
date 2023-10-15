from hezar import Model


model = Model.load("hezarai/t5-base-fa")
text = "سلام حاجی چه خبر"
outputs = model.predict(text)
print(outputs)
