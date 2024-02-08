# Installation

## Install from PyPi
Installing Hezar is as easy as any other Python library! Most of the requirements are cross-platform and installing
them on any machine is a piece of cake!

```
pip install hezar
```
### Installation variations
Hezar is packed with a lot of tools that are dependent on other packages. Most of the
time you might not want everything to be installed, hence, providing multiple variations of
Hezar so that the installation is light and fast for general use.

You can install optional dependencies for each mode like so:
```
pip install hezar[nlp]  # For natural language processing
pip install hezar[vision]  # For computer vision and image processing
pip install hezar[audio]  # For audio and speech processing
pip install hezar[embeddings]  # For word embeddings
```
Or you can also install everything using:
```
pip install hezar[all]
```
## Install from source
Also, you can install the dev version of the library using the source:
```
pip install git+https://github.com/hezarai/hezar.git
```

## Test installation
From a Python console or in CLI just import `hezar` and check the version:
```python
import hezar

print(hezar.__version__)
```
```
0.23.1
```
