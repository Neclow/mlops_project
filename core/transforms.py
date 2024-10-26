"""Data transforms"""

import random

import torch

from torchaudio.transforms import Vad


class Pad(torch.nn.Module):
    def __init__(self, sr, max_duration, direction="random"):
        """
        Audio zero-padding

        Parameters
        ----------
        sr : int
            Sample rate
        max_duration : float
            Max audio duration (in s)
        direction : str, optional
            Padding direction, by default "random"
        """
        super().__init__()
        self.sr = sr
        self.max_duration = max_duration
        self.max_length = int(self.sr * self.max_duration)

        assert direction in ("left", "right", "random")
        self.direction = direction

    def forward(self, x):
        sample_length = len(x)

        pad_length = self.max_length - sample_length

        if pad_length > 0:
            padding = torch.zeros(pad_length, dtype=x.dtype)

            if self.direction == "random":
                direction = "left" if random.random() < 0.5 else "right"
            else:
                direction = self.direction

            if direction == "left":
                x = torch.cat((padding, x), dim=0)
            else:
                x = torch.cat((x, padding), dim=0)
        return x


class Trim(torch.nn.Module):
    """
    Audio trimming

    Parameters
    ----------
    sr : int
        Sample rate
    max_duration : float
        Max audio duration (in s)
    direction : str, optional
        Trimming direction, by default "random"
    """

    def __init__(self, sr, max_duration, direction="random"):
        super().__init__()
        self.sr = sr
        self.max_duration = max_duration
        self.max_length = int(self.sr * self.max_duration)

        assert direction in ("left", "right", "random")
        self.direction = direction

    def forward(self, x):
        sample_length = len(x)

        trim_length = sample_length - self.max_length - 1

        if trim_length > 0:
            if self.direction == "random":
                start = random.randint(0, trim_length)
                stop = start + self.max_length
            elif self.direction == "left":
                start = 0
                stop = self.max_length
            else:
                start = trim_length + 1
                stop = sample_length

            x = x[start:stop]

        return x


def load_transforms(sr, max_duration, vad=False):
    transforms = []

    if vad:
        transforms.append(Vad(sample_rate=sr))

    transforms.extend(
        [
            Trim(
                sr=sr,
                max_duration=max_duration,
            ),
            Pad(
                sr=sr,
                max_duration=max_duration,
                direction="right",
            ),
        ]
    )

    return torch.nn.Sequential(*transforms)
