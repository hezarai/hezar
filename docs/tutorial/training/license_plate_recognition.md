# Image to Text (License Plate Recognition)
Image to text is the task of generating text from an image e.g, image captioning, OCR, document understanding, etc.
In Hezar, the `image2text` task is responsible for all of those which currently includes image captioning and OCR.

In this tutorial, we'll finetune a base OCR model (CRNN) on a license plate recognition dataset.

## Dataset
In Hezar, there are two types of `image2text` datasets: `OCRDataset` and `ImageCaptioningDataset`. The reason is that
depending on the model, the dataset labels structure is different; For example, `CRNN` requires the labels as characters
but Transformer-based models like `ViTRoberta` requires the labels as token ids. So here, we'll use the `OCRDataset` class.

### Hezar ALPR Dataset
We do provide a pretty solid ALPR dataset at [hezarai/persian-license-plate-v1](https://huggingface.co/datasets/hezarai/persian-license-plate-v1)
which you can load as easily as:
```python
from hezar.data import Dataset

max_length = 8
reverse_digits = True

train_dataset = Dataset.load("hezarai/persian-license-plate-v1", split="train")
eval_dataset = Dataset.load("hezarai/persian-license-plate-v1", split="test")
```
- License plates have only 8 characters so we set the max_length=8 which makes the dataset remove longer/shorter samples
- CRNN's image processor flips the image horizontally (mirror) for Persian but since plates are read in LTR mode we have to set 
the `reverse_digits=True` so that the labels are represented in RTL mode.

That's all we need to do to create our datasets ready for passing to the Trainer. If you need to create your own dataset
you can also see the next section or skip it otherwise.

### Custom ALPR Dataset
Let's see how to create a custom dataset for OCR. When it comes to customizing a dataset with a supported task in Hezar,
there are two ways in general; Subclassing the dataset class of that task in particular and subclassing the base `Dataset`
class. 

Since we're customizing an `image2text` dataset, we can override the `OCRDataset` class.

Let's consider you have a CSV file of your dataset with two columns: `image_path`, `text`.

```python
from hezar.data import OCRDataset, OCRDatasetConfig


class ALPRDataset(OCRDataset):
    def __init__(self, config: OCRDatasetConfig, split=None, **kwargs):
        super().__init__(config=config, split=split, **kwargs)

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
from hezar.models import CRNNImage2TextConfig, CRNNImage2Text
from hezar.preprocessors import Preprocessor

base_model_path = "hezarai/crnn-fa-printed-96-long"

model = CRNNImage2Text(CRNNImage2TextConfig(id2label=train_dataset.config.id2label))
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
Incompatible keys: ['map2seq.weight', 'map2seq.bias', 'rnn1.weight_ih_l0', 'rnn1.weight_ih_l0_reverse', 'classifier.weight', 'classifier.bias']
Missing keys: []


******************** Training Info ********************

  Output Directory: crnn-plate-fa-v2
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
  Number of Parameters: 9254637
  Number of Trainable Parameters: 9254637
  Mixed Precision: Full (fp32)
  Gradient Accumulation Steps: 1
  Metrics: ['cer']
  Save Steps: 991
  Checkpoints Path: crnn-plate-fa-v2/checkpoints
  Logs Path: crnn-plate-fa-v2/logs/Jun09_11-56-57_Taycan

*******************************************************


Epoch: 1/10     100%|######################################################################| 991/991 [01:07<00:00, 14.70batch/s, loss=3.5] 
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 17.24batch/s, cer=1, loss=2.81]

Epoch: 2/10     100%|######################################################################| 991/991 [01:07<00:00, 14.65batch/s, loss=2.97]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 17.08batch/s, cer=1, loss=1.87]

Epoch: 3/10     100%|######################################################################| 991/991 [01:08<00:00, 14.49batch/s, loss=2.41]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 16.73batch/s, cer=0.196, loss=0.708]

Epoch: 4/10     100%|######################################################################| 991/991 [01:07<00:00, 14.59batch/s, loss=1.93]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 16.83batch/s, cer=0.0883, loss=0.323]

Epoch: 5/10     100%|######################################################################| 991/991 [01:08<00:00, 14.46batch/s, loss=1.59]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 16.62batch/s, cer=0.0405, loss=0.193]

Epoch: 6/10     100%|######################################################################| 991/991 [01:10<00:00, 14.15batch/s, loss=1.35]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 16.62batch/s, cer=0.0335, loss=0.165]

Epoch: 7/10     100%|######################################################################| 991/991 [01:08<00:00, 14.48batch/s, loss=1.17]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 16.82batch/s, cer=0.0301, loss=0.132]

Epoch: 8/10     100%|######################################################################| 991/991 [01:09<00:00, 14.29batch/s, loss=1.03]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 16.94batch/s, cer=0.0285, loss=0.13] 

Epoch: 9/10     100%|######################################################################| 991/991 [01:08<00:00, 14.38batch/s, loss=0.926]
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 17.24batch/s, cer=0.0316, loss=0.136]

Epoch: 10/10    100%|######################################################################| 991/991 [01:09<00:00, 14.25batch/s, loss=0.84] 
Evaluating...   100%|######################################################################| 125/125 [00:07<00:00, 16.49batch/s, cer=0.0289, loss=0.123]
Hezar (INFO): Training done!
```
In less than 10 minutes on a cheap GTX 1660 GPU, our model acheives a character error rate as low as 2 percent! 
A few more epochs will indeed give better results but that's up to you.
