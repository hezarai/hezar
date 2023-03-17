from hezar import Model


model = Model.load("hezar-ai/mlp-test")
inputs = [1, 2, 3, 4]
outputs = model.predict(inputs)
print(outputs)
