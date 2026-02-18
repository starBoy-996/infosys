 
"""
def audio_to_embedding(audio):
    embedding = [0.12, 0.34]  # dummy example
    return embedding

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Load once (global)
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()


def audio_to_embedding(audio_path: str):
    # Load audio
     
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz (required)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)

    # Process
    inputs = processor(
        waveform.squeeze(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling → fixed vector
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding.tolist()

import torchaudio


import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

import torchaudio
 

import torch
 


from transformers import Wav2Vec2Processor, Wav2Vec2Model


 

 
 


import os
os.system("pip install soundfile")


  


import soundfile as sf
import torch

waveform, sr = sf.read(audio_path)
waveform = torch.tensor(waveform).float()

if waveform.ndim == 1:
    waveform = waveform.unsqueeze(0)


import torchaudio
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
model.eval()

def audio_to_embedding(audio_path: str):
    waveform, sample_rate = torchaudio.load(audio_path)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, 16000
        )

    inputs = processor(
        waveform.squeeze(),
        sampling_rate=16000,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding.tolist()
"""

 

 


import torch
import torchaudio
from transformers import Wav2Vec2Model

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
model.eval()

def audio_to_embedding(audio_path: str):
    waveform, sample_rate = torchaudio.load(
        audio_path,
        backend="soundfile"
    )

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, 16000
        )

    with torch.no_grad():
        outputs = model(waveform)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings


 
# force backend to soundfile, which is stable


waveform, sample_rate = torchaudio.load("data/user_audio/test.wav")


 

 

import torchaudio

# make sure the variable matches
audio_path = "data/user_audio/she.wav"

# optionally, force soundfile backend to avoid TorchCodec if needed
torchaudio.set_audio_backend("soundfile")

waveform, sample_rate = torchaudio.load(audio_path)


