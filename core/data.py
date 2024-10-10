"""Dataset classes"""

import json
import os
import sys

from abc import abstractmethod
from glob import glob
from pathlib import Path

from speechbrain.dataio.encoder import CategoricalEncoder
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset for this project

    Parameters
    ----------
    dataset : str
        Name of a dataset. Example: `fleurs`.
    subset : str
        Data subset (`train`, `dev`, `test`).
    root_dir : str
        Root folder path for analyses
    save_dir : str, optional
        Path to save artefactsl, by default "data"
    transform : callable, optional
        Optional transform to be applied on a sample, by default None
    target_transform : callable, optional
        Optional transform to be applied on a label, by default None
    """

    def __init__(
        self,
        dataset,
        subset,
        root_dir,
        save_dir="data",
        transform=None,
        target_transform=None,
    ):
        self.dataset = dataset
        self.subset = subset
        self.root_dir = root_dir

        self.data_folder = f"{self.root_dir}/datasets/{self.dataset}"

        with open(
            f"{self.root_dir}/languages/{self.dataset}.json", "r", encoding="utf-8"
        ) as f:
            languages = json.load(f)

        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        self.label_encoder = CategoricalEncoder()
        self.label_encoder.load_or_create(
            path=os.path.join(self.save_dir, f"{self.dataset}_lang_encoder.txt"),
            from_iterables=[list(languages.keys())],
            output_key="lang_id",
        )

        self.transform = transform
        self.target_transform = target_transform

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError


class AudioDataset(BaseDataset):
    """Dataset for audio samples

    Parameters
    ----------
    processor : core.upstream.AudioProcessor
        an object with a ```process``` function
    max_duration : float, optional
        Maximum audio duration, by default None
    ext : str, optional
        Audio file extension, by default "wav"
    """

    def __init__(
        self,
        processor,
        max_duration=None,
        ext="wav",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.processor = processor

        if max_duration is None:
            self.max_length = sys.maxsize
        else:
            self.max_length = int(max_duration * self.processor.sr)

        self.ext = ext
        self.audio_file_list = glob(
            f"{self.data_folder}/*/*/audio/{self.subset}/*.{ext}"
        )

    def __len__(self):
        return len(self.audio_file_list)

    def __getitem__(self, idx):
        # Path: root_dir/datasets/name_of_dataset/language/language/audio/subset/*.wav
        wav_path = self.audio_file_list[idx]

        x = self.processor(wav_path)

        language = Path(wav_path).parents[3].stem
        label = self.label_encoder.encode_label(language)

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            label = self.target_transform(label)

        return {"input": x, "label": label}


def load_data(dataset, root_dir, save_dir, **kwargs):
    """Load audio data

    Parameters
    ----------
    dataset : str
        Name of a dataset. Example: `fleurs`.
    root_dir : str
        Root folder path for analyses
    save_dir : str, optional
        Path to save artefacts

    Returns
    -------
    train_dataset : torch.utils.data.Dataset object
        Train dataset
    valid_dataset : torch.utils.data.Dataset object
        Validation dataset
    test_dataset : torch.utils.data.Dataset object
        Test dataset
    """
    train_dataset = AudioDataset(
        subset="train",
        dataset=dataset,
        root_dir=root_dir,
        save_dir=save_dir,
        **kwargs,
    )

    valid_dataset = AudioDataset(
        subset="dev",
        dataset=dataset,
        root_dir=root_dir,
        save_dir=save_dir,
        **kwargs,
    )

    test_dataset = AudioDataset(
        subset="test",
        dataset=dataset,
        root_dir=root_dir,
        save_dir=save_dir,
        **kwargs,
    )

    return train_dataset, valid_dataset, test_dataset
