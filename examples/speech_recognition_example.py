from hezar import Model

whisper = Model.load("hezarai/whisper-small-fa")
transcripts = whisper.predict("assets/speech_example.mp3")
print(transcripts)
