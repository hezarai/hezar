from hezar.models import Model


path = "hezarai/roberta-fa-mask-filling"
text = ["سلام بچه ها حالتون <mask>"]
model = Model.load(path)
outputs = model.predict(text, top_k=1, device="cuda")
print(outputs)
