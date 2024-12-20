"""Dataset classes"""

import os
import sys

from abc import abstractmethod
from glob import glob

import pandas as pd

from speechbrain.dataio.encoder import CategoricalEncoder
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base dataset for this project

    Parameters
    ----------
    dataset : str
        Name of a dataset. Example: `fleurs`.
    subset : str
        Data subset (`train`, `dev`, `test`).
    data_dir : str
        Path to main data folder path
    transform : callable, optional
        Optional transform to be applied on a sample, by default None
    target_transform : callable, optional
        Optional transform to be applied on a label, by default None
    """

    def __init__(
        self,
        dataset,
        subset,
        data_dir="data",
        transform=None,
        target_transform=None,
    ):
        # Base parameters
        self.dataset = dataset
        self.subset = subset
        self.data_dir = data_dir

        # Language metadata
        self.language_metadata = pd.read_csv(
            f"{self.data_dir}/languages/{self.dataset}.csv"
        )
        languages = self.language_metadata.language.to_list()

        # Label encoding
        self.label_dir = f"{self.data_dir}/labels"
        os.makedirs(self.label_dir, exist_ok=True)
        self.label_encoder = CategoricalEncoder()
        self.label_encoder.load_or_create(
            path=os.path.join(self.label_dir, f"{self.dataset}.txt"),
            from_iterables=[languages],
            output_key="lang_id",
        )

        # Data transforms
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
    max_duration : float
        Maximum audio duration in seconds
    ext : tuple, optional
        Allowed audio file extensions, by default ("wav", "mp3")
    """

    def __init__(
        self,
        processor,
        max_duration=None,
        exts=("wav", "mp3"),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.processor = processor

        if max_duration is None:
            self.max_length = sys.maxsize
        else:
            self.max_length = int(max_duration * self.processor.sr)

        base_pattern = f"{self.data_dir}/{self.dataset}/**/audio/{self.subset}"

        self.audio_file_list = []

        for ext in exts:
            self.audio_file_list.extend(glob(f"{base_pattern}/*.{ext}", recursive=True))

        if len(self.audio_file_list) == 0:
            raise FileNotFoundError("No files found. Check `self.data_dir`.")

    def __len__(self):
        return len(self.audio_file_list)

    def __getitem__(self, idx):
        # pattern: data_dir/dataset/subdataset/language/language/audio/subset/fname.ext
        wav_path = self.audio_file_list[idx]

        x = self.processor(wav_path)

        # pylint: disable=unused-variable
        subdataset, language_code = wav_path.replace(
            f"{self.data_dir}/{self.dataset}/", ""
        ).split(os.sep)[:2]

        language = self.language_metadata.query(
            f"`{subdataset}` == @language_code"
        ).language.item()
        # pylint: enable=unused-variable

        label = self.label_encoder.encode_label(language)

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            label = self.target_transform(label)

        return {"input": x, "label": label}


def load_data(dataset, data_dir, **kwargs):
    """
    Load audio data

    Parameters
    ----------
    dataset : str
        Name of a dataset. Example: `fleurs`.
    data_dir : str
        Path to main data folder path
    **kwargs
        Optional arguments passed to AudioDataset

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
        data_dir=data_dir,
        **kwargs,
    )

    valid_dataset = AudioDataset(
        subset="dev",
        dataset=dataset,
        data_dir=data_dir,
        **kwargs,
    )

    test_dataset = AudioDataset(
        subset="test",
        dataset=dataset,
        data_dir=data_dir,
        **kwargs,
    )

    return train_dataset, valid_dataset, test_dataset
