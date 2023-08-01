from hezar import Model

model = Model.load("t5-base-fa")
text = "سلام حاجی چه خبر"
outputs = model.predict(text)
print(outputs)
