import random

import torch


class Pad(torch.nn.Module):
    def __init__(self, sample_rate, max_duration, direction="random"):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_length = int(self.sample_rate * self.max_duration)

        # TODO: add bidirectional padding mode
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
    def __init__(self, sample_rate, max_duration, direction="random"):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_length = int(self.sample_rate * self.max_duration)

        # TODO: add bidirectional trimming mode
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
