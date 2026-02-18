import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def audio_to_embedding(audio_path):
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

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding
