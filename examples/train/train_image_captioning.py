from hezar.models import ViTRobertaImage2Text, ViTRobertaImage2TextConfig, build_vision_encoder_decoder_model, Model
from hezar.preprocessors import Preprocessor, ImageProcessor, ImageProcessorConfig
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig
from hezar.utils import get_state_dict_from_hub


base_decoder_path = "hezarai/roberta-base-fa"
base_encoder_path = "vit-base-patch16-224"

train_dataset = Dataset.load("hezarai/flickr30k-fa", split="train")
eval_dataset = Dataset.load("hezarai/flickr30k-fa", split="test")

model = ViTRobertaImage2Text(ViTRobertaImage2TextConfig())
encoder = Model.load(base_encoder_path)
decoder = Model.load(base_decoder_path)
vision_encoder_decoder = build_vision_encoder_decoder_model(encoder, decoder)

model.load_state_dict(vision_encoder_decoder.state_dict())

tokenizer = Preprocessor.load(base_decoder_path)
image_processor = ImageProcessor(
    ImageProcessorConfig(
        mean=[0.5],
        std=[0.5],
        rescale=1/255,
        resample=2,
        size=(224, 224),
    )
)
model.preprocessor = [tokenizer, image_processor]


train_config = TrainerConfig(
    output_dir="crnn-plate-fa-v1",
    task="image2text",
    device="cuda",
    batch_size=8,
    num_epochs=20,
    metrics=["cer"],
    metric_for_best_model="cer"
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
)
trainer.train()
