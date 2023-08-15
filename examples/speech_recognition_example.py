from hezar import Model
from datasets import load_dataset, Audio

ds = load_dataset("mozilla-foundation/common_voice_11_0", "fa", split="test")
df = ds.cast_column("audio", Audio(sampling_rate=16000))
sample = ds[1001]
whisper = Model.load("hezarai/whisper-small-fa")
transcript = whisper.predict(sample["path"])  # or pass `sample["audio"]["array"]` (with the right sample rate)
print(transcript)
