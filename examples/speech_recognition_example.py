from hezar import Model
from datasets import load_dataset

ds = load_dataset("mozilla-foundation/common_voice_11_0", "fa", split="test")
sample = ds[1001]
whisper = Model.load("hezarai/whisper-small-fa")
transcript = whisper.predict(sample["path"])  # or pass `sample["audio"]["array"]` (with the right sample rate)
print(transcript)
