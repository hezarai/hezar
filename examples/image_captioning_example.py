from hezar import Model


model = Model.load("hezarai/vit-roberta-fa-image-captioning-flickr30k")
texts = model.predict("assets/image_captioning_example.jpg")
print(texts)
