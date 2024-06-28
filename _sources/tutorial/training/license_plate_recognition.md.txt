# Image to Text (License Plate Recognition)
Image to text is the task of generating text from an image e.g, image captioning, OCR, document understanding, etc.
In Hezar, the `image2text` task is responsible for all of those which currently includes image captioning and OCR.

In this tutorial, we'll finetune a base OCR model (CRNN) on a license plate recognition dataset.
```python
from hezar.data import Dataset
from hezar.models import CRNNImage2TextConfig, CRNNImage2Text
from hezar.preprocessors import Preprocessor

base_model_path = "hezarai/crnn-fa-printed-96-long"
```

## Dataset
In Hezar, there are two types of `image2text` datasets: `OCRDataset` and `ImageCaptioningDataset`. The reason is that
depending on the model, the dataset labels structure is different; For example, `CRNN` requires the labels as characters
but Transformer-based models like `ViTRoberta` requires the labels as token ids. So here, we'll use the `OCRDataset` class.

### Option 1: Hezar ALPR Dataset
We do provide a pretty solid ALPR dataset at [hezarai/persian-license-plate-v1](https://huggingface.co/datasets/hezarai/persian-license-plate-v1)
which you can load as easily as:
```python
max_length = 8
reverse_digits = True

train_dataset = Dataset.load(
    "hezarai/persian-license-plate-v1",
    split="train",
    preprocessor=base_model_path,
    max_length=8,
    reverse_digits=True,
)
eval_dataset = Dataset.load(
    "hezarai/persian-license-plate-v1",
    split="test",
    preprocessor=base_model_path,
    max_length=8,
    reverse_digits=True,
)
```
- License plates have only 8 characters so we set the max_length=8 which makes the dataset remove longer/shorter samples
- CRNN's image processor flips the image horizontally (mirror) for Persian but since plates are read in LTR mode we have to set 
the `reverse_digits=True` so that the labels are represented in RTL mode.
- For the `preprocessor` we use the base model's preprocessor which is an `ImageProcessor` instance to do the job.

That's all we need to do to create our datasets ready for passing to the Trainer. If you need to create your own dataset
you can also see the next section or skip it otherwise.

### Option 2: Custom ALPR Dataset
Let's see how to create a custom dataset for OCR. When it comes to customizing a dataset with a supported task in Hezar,
there are two ways in general; Subclassing the dataset class of that task in particular and subclassing the base `Dataset`
class. 

Since we're customizing an `image2text` dataset, we can override the `OCRDataset` class.

Let's consider you have a CSV file of your dataset with two columns: `image_path`, `text`.

```python
import pandas as pd
from hezar.data import OCRDataset, OCRDatasetConfig


class ALPRDataset(OCRDataset):
    def __init__(self, config: OCRDatasetConfig, split=None, preprocessor=None, **kwargs):
        super().__init__(config=config, split=split, preprocessor=preprocessor, **kwargs)

    # Override the `_load` method (originally loads a dataset from the Hub) to load the csv file
    def _load(self, split=None):
        # Load a dataframe here and make sure the split is fetched
        data = pd.read_csv(self.config.path)
        # Configure id2label from the dataset
        unique_characters = sorted(set("".join(data["text"])))
        self.config.id2label = {idx: c for idx, c in enumerate(unique_characters)}
        # preprocess if needed
        return data

    def __getitem__(self, index):
        path, text = self.data.iloc[index].values()
        # The `image_processor` (`ImageProcessor`) loads the image file and processes it base on it's config
        pixel_values = self.image_processor(path, return_tensors="pt")["pixel_values"][0]
        # The `_text_to_tensor` converts the raw text to a tensor using the `id2label` attribute.
        labels = self._text_to_tensor(text)
        inputs = {
            "pixel_values": pixel_values,
            "labels": labels,
        }
        return inputs
```

You can customize this class further according to your needs.

## Model
For the model we'll use the `CRNN` model with pretrained weights from `hezarai/crnn-fa-printed-96-long` which was trained
on a large Persian corpus with millions of synthetic samples. 
```python
model_config = CRNNImage2TextConfig.load(base_model_path, id2label=train_dataset.config.id2label)
model = CRNNImage2Text(model_config)
preprocessor = Preprocessor.load(base_model_path)
```
- The preprocessor is the same as the base model so we set the preprocessor to that.
- We use the `id2label` from our dataset config

## Training
Now everything's ready to start the training.

```python
train_config = TrainerConfig(
    output_dir="crnn-plate-fa",
    task="image2text",
    device="cuda",
    init_weights_from=base_model_path,
    resume_from_checkpoint=True,
    batch_size=8,
    num_epochs=10,
    log_steps=100,
    metrics=["cer"],
    metric_for_best_model="cer"
)

trainer = Trainer(
    config=train_config,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=train_dataset.data_collator,
    preprocessor=preprocessor,
)
```
- We initialize the weights of the model from the pretrained path set earlier.
- We use the CER (Character Error Rate) metric to evaluate the model (lower is better, 0.05 is fine, 0.02 and below is awesome!).
- Resuming from the last checkpoint is also good to be set so that the training is continued if interrupted.

Let's start the training...
```python
trainer.train()
```
```
Hezar (WARNING): 37 invalid samples found in the dataset! Inspect them using the `invalid_data` attribute
Hezar (WARNING): 2 invalid samples found in the dataset! Inspect them using the `invalid_data` attribute
Hezar (WARNING): Partially loading the weights as the model architecture and the given state dict are incompatible! 
Ignore this warning in case you plan on fine-tuning this model
Incompatible keys: ['classifier.weight', 'classifier.bias']
Missing keys: []


******************** Training Info ********************

  Output Directory: crnn-plate-fa
  Task: image2text
  Model: CRNNImage2Text
  Init Weights: hezarai/crnn-fa-printed-96-long
  Device(s): cuda
  Batch Size: 8
  Epochs: 10
  Total Steps: 9910
  Training Dataset: OCRDataset(path=hezarai/persian-license-plate-v1['train'], size=7925)
  Evaluation Dataset: OCRDataset(path=hezarai/persian-license-plate-v1['test'], size=993)
  Optimizer: adam
  Scheduler: None
  Initial Learning Rate: 2e-05
  Learning Rate Decay: 0.0
  Number of Parameters: 9287437
  Number of Trainable Parameters: 9287437
  Mixed Precision: Full (fp32)
  Gradient Accumulation Steps: 1
  Metrics: ['cer']
  Save Steps: 991
  Log Steps: 100
  Checkpoints Path: crnn-plate-fa/checkpoints
  Logs Path: crnn-plate-fa/logs/Jun09_13-23-16_Taycan

*******************************************************


Epoch: 1/10     100%|######################################################################| 991/991 [01:07<00:00, 14.59batch/s, loss=3.55]
Evaluating...   100%|######################################################################| 125/125 [00:08<00:00, 15.32batch/s, cer=1, loss=2.89]

Epoch: 2/10     100%|######################################################################| 991/991 [01:09<00:00, 14.31batch/s, loss=3.12]
Evaluating...   100%|######################################################################| 125/125 [00:08<00:00, 15.17batch/s, cer=1, loss=2.28]

Epoch: 3/10     100%|######################################################################| 991/991 [01:09<00:00, 14.24batch/s, loss=2.63]
Evaluating...   100%|######################################################################| 125/125 [00:08<00:00, 15.16batch/s, cer=0.338, loss=1.01]

Epoch: 4/10     100%|######################################################################| 991/991 [01:09<00:00, 14.26batch/s, loss=2.14]
Evaluating...   100%|######################################################################| 125/125 [00:08<00:00, 15.11batch/s, cer=0.141, loss=0.452]

Epoch: 5/10     100%|######################################################################| 991/991 [01:09<00:00, 14.23batch/s, loss=1.78]
Evaluating...   100%|######################################################################| 125/125 [00:08<00:00, 15.09batch/s, cer=0.0644, loss=0.286]

Epoch: 6/10     100%|######################################################################| 991/991 [01:09<00:00, 14.23batch/s, loss=1.52]
Evaluating...   100%|######################################################################| 125/125 [00:08<00:00, 15.10batch/s, cer=0.056, loss=0.241] 

Epoch: 7/10     100%|######################################################################| 991/991 [01:08<00:00, 14.52batch/s, loss=1.32]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 15.92batch/s, cer=0.0521, loss=0.217]

Epoch: 8/10     100%|######################################################################| 991/991 [01:06<00:00, 14.82batch/s, loss=1.17]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 15.79batch/s, cer=0.0548, loss=0.22] 

Epoch: 9/10     100%|######################################################################| 991/991 [01:06<00:00, 14.82batch/s, loss=1.05]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 15.85batch/s, cer=0.0481, loss=0.197]

Epoch: 10/10    100%|######################################################################| 991/991 [01:06<00:00, 14.81batch/s, loss=0.953]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 15.88batch/s, cer=0.0465, loss=0.189]
Hezar (INFO): Training done!
```
In less than 10 minutes on a cheap GTX 1660 GPU, our model acheives a character error rate as low as 4 percent! 
A few more epochs will indeed give better results but that's up to you.

## Push to Hub
If you'd like, you can push the model along with other Trainer files to the Hub.
```python
trainer.push_to_hub("<path/to/model>", commit_message="Upload an awesome ALPR model!")
```