"""Audio loaders/processors"""

import librosa
import torch
import torchaudio
import whisper

from speechbrain.dataio.preprocess import AudioNormalizer
from transformers import AutoFeatureExtractor


class AudioProcessor:
    def __init__(self, sr):
        self.sr = sr

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def process(self, wav_path):
        x, _ = librosa.load(wav_path, sr=self.sr)

        return torch.from_numpy(x)


class HuggingfaceProcessor(AudioProcessor):
    def __init__(self, model_id, sr, cache_dir=None):
        super().__init__(sr)

        self.model_id = model_id

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/huggingface"

        self.processor = AutoFeatureExtractor.from_pretrained(
            self.model_id, cache_dir=cache_dir
        )

    def process(self, wav_path):
        x = super().process(wav_path)

        x = self.processor(x, sampling_rate=self.sr, return_tensors="pt")

        x = x.input_values[0]

        return x


class SpeechbrainProcessor(AudioProcessor):
    def __init__(self, sr):
        super().__init__(sr)

        self.audio_normalizer = AudioNormalizer(sample_rate=self.sr)

    def process(self, wav_path):
        x, sr = torchaudio.load(wav_path, channels_first=False)

        x = self.audio_normalizer(x, sr)

        return x


class WhisperProcessor(AudioProcessor):
    def process(self, wav_path):
        x = whisper.load_audio(wav_path, sr=self.sr)

        return torch.from_numpy(x)


def load_processor(model_id, sr, cache_dir=None) -> AudioProcessor:
    """
    Load an audio data processor (to be used in AudioDataset)

    (Audio normalisation, spectrograms...)

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    sr : int
        Sample rate
    cache_dir : str, optional
        Path to the folder where cached files are stored, by default None

    Returns
    -------
    AudioProcessor
        an object with a ```process``` function
    """
    if any(_ in model_id for _ in ["xls-r", "mms"]):
        processor = HuggingfaceProcessor(model_id, sr=sr, cache_dir=cache_dir)
    elif "speechbrain" in model_id:
        processor = SpeechbrainProcessor(sr=sr)
    elif "whisper" in model_id:
        processor = WhisperProcessor(sr)
    else:
        processor = AudioProcessor(sr)

    return processor
