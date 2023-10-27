# Add a Dataset
Adding datasets involves two main steps:
1. Uploading the dataset to the Hub and providing a load script.
2. Providing a proper dataset class in Hezar.

## Uploading dataset to the Hub
Datasets of different types, require different format in terms of raw files and annotations. In Hezar, we prefer
uploading the files to the same repo of the dataset. The way the datasets are provided on the Hub is really up to you,
but conventionally, it's better to follow the same procedure for every dataset. The recommended way is to get
inspiration from other datasets on the Hub that have a similar task. Either datasets provided by Hezar or others.
Some notes to consider:
- Providing zip files rather that folder of raw files is recommended.
- For datasets containing raw files like images, audio files, etc. use a csv annotation file mapping files to labels.
- Providing both train and test splits is a must, but validation set is optional.
- Put all files in the `data` folder and put `X_train.zip`, `X_test.zip`, `X_validation.zip` inside it or put all files named after splits inside a `data.zip` file.
- Don't forget to provide a dataset card (`README.md`) and specify properties such as task, license, tags, etc.

## Providing a loading script
Hezar has some ready to use templates for dataset loading scripts. You can find them [here](https://github.com/hezarai/hezar/tree/main/templates/dataset_scripts).
You can learn more about dataset loading scripts [here](https://huggingface.co/docs/datasets/dataset_script).
It's recommended to upload the dataset to the Hub to test that it works properly.
