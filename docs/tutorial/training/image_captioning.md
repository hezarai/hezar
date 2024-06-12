# Image to Text (Image Captioning)

Image captioning is the task of generating text from a given image. Mostly used for describing what's going on in an
image. In this tutorial, we'll train a encoder-decoder based (ViT+RoBERTa) model on the Flickr30K dataset (Persian).

Let's first import everything needed.
```python
from hezar.models import Model
from hezar.data import Dataset
from hezar.trainer import Trainer, TrainerConfig
```
We'll use the base `ViTRobertaImage2Text` model with a pretrained weights like below:
```python
base_model_path = "hezarai/vit-roberta-fa-base"
```

## Dataset

### Option 1: Persian Flickr30K 
The flickr30k dataset is already hosted and ready to use in Hezar's Hugging Face Hub.

```python
train_dataset = Dataset.load("hezarai/flickr30k-fa", split="train", preprocessor=base_model_path, max_length=128)
eval_dataset = Dataset.load("hezarai/flickr30k-fa", split="test", preprocessor=base_model_path, max_length=128)
```
### Option 2: Custom Image Captioning Dataset
Let's see how we can create a custom dataset for this task. When it comes to customizing a dataset with a supported task in Hezar,
there are two ways in general; Subclassing the dataset class of that task in particular and subclassing the base `Dataset`
class. 

Since we're customizing an `image2text` dataset, we can override the `ImageCaptioningDataset` class.

Let's consider you have a CSV file of your dataset with two columns: `image_path`, `text`.

```python
import torch
import pandas as pd

from hezar.data import ImageCaptioningDataset, ImageCaptioningDatasetConfig
from hezar.utils import shift_tokens_right


class Flickr30kDataset(ImageCaptioningDataset):
    def __init__(self, config: ImageCaptioningDatasetConfig, split=None, **kwargs):
        super().__init__(config=config, split=split, **kwargs)

    # Override the `_load` method (originally loads a dataset from the Hub) to load the csv file
    def _load(self, split=None):
        # Load a dataframe here and make sure the split is fetched
        data = pd.read_csv(self.config.path)
        # preprocess if needed
        return data

    def __getitem__(self, index):
        path, text = self.data.iloc[index].values()
        # The `image_processor` (`ImageProcessor`) loads the image file and processes it base on it's config
        pixel_values = self.image_processor(path, return_tensors="pt")["pixel_values"]
        tokenized_inputs = self.tokenizer(text, padding="max_length", max_length=self.config.max_length)
        labels = torch.tensor([tokenized_inputs["token_ids"]])
        attention_mask = torch.tensor([tokenized_inputs["attention_mask"]])
        decoder_input_ids = shift_tokens_right(
            labels,
            pad_token_id=self.tokenizer.pad_token_id,
            decoder_start_token_id=self.tokenizer.bos_token_id,
        )
        inputs = {
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": attention_mask,
        }
        return inputs
```

You can customize this class further according to your needs.

### Data Collator
The default data collator of the `ImageCaptioningDataset` named `ImageCaptioningDataCollator` handles the data collation
by padding the tokenizer outputs based on max length and padding type.

## Model
For the model we'll use the `ViTRobertaImage2Text` model with pretrained weights from `hezarai/vit-roberta-fa-base` 
which was created by joining the `hezarai/roberta-base-fa` and `google/vit-base-patch16-224-in21k` and integrated in Hezar.
This joint, introduces some additional layers and parameters which will be trained from scratch but the rest of the
weights are pretrained.

```python
model = Model.load(base_model_path)
```
This will load the model weights and the preprocessor.


## Training
Now everything's ready to start the training.

```python
train_config = TrainerConfig(
    output_dir="vit-roberta-fa-image-captioning-flickr30k",
    task="image2text",
    device="cuda",
    batch_size=12,
    num_epochs=20,
    mixed_precision="fp16",
    resume_from_checkpoint=True,
    log_steps=100,
    save_steps=500,
    metrics=["wer"],
    metric_for_best_model="wer"
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
)
```
- Since this is a big model and the training might take a while, it's recommended to use mixed precision and saving on every 500 steps. 
- For evaluation, we'll use the WER (Word Error Rate).

Now let's run the training!
```python
trainer.train()
```

## Push to Hub
If you'd like, you can push the model along with other Trainer files to the Hub.
```python
trainer.push_to_hub("<path/to/model>", commit_message="Upload an awesome image captioning model!")
```