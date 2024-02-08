from hezar.models import Model

example = ["هزار، کتابخانه‌ای کامل برای به کارگیری آسان هوش مصنوعی"]
model = Model.load("hezarai/roberta-fa-sentiment-dksf")
outputs = model.predict(example)
print(outputs)
