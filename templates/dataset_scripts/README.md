# Datasets Loading Scripts Templates (DEPRECATED)
In this section you can explore different ready to use templates for writing a dataset script.
These scripts are separated by dataset file type (csv, parquet, etc) and task (text classification, etc).

### Write your own script
First choose the desired template based on your dataset file format and task.
For example if you have a sentiment analysis dataset in CSV format, you have to choose
`csv/text_classification.py` file. In this file all the instructions are documented for every part.
Some are mandatory and some are optional and are specified by a `# TODO` comment around them.

For an in-depth documentation on how datasets loading scripts work in the HuggingFace Hub, read [here](https://huggingface.co/docs/datasets/dataset_script).

#### Step 1
Grab the template and apply all the required changes to it.
#### Step 2
In another file or console, you can test load the script using this code:
```python
from datasets import load_dataset

dataset = load_dataset(path="path/to/loading/script.py", split="train")
print(dataset[0])

```
