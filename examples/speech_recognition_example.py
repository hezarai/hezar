from hezar import Model
import numpy as np


whisper = Model.load("hezarai/whisper-small")
audio_array = np.load("sample_audio.npy")
transcript = whisper.predict([audio_array])
print(transcript)
