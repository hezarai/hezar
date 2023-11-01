from hezar.models import Model


model = Model.load("hezarai/gpt2-base-fa")
text = "در سال‌های اخیر با پیشرفت هوش مصنوعی "
outputs = model.predict(text)
print(outputs)
