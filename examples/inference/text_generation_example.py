from hezar.models import Model


model = Model.load("hezarai/gpt2-base-fa")
text = "سلام حاجی چه خبر"
outputs = model.predict(text)
print(outputs)
