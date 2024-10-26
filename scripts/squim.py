"""
Reference-free speech quality assessment using torchaudio SQUIM

Adapted from https://pytorch.org/audio/main/tutorials/squim_tutorial.html
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob

import pandas as pd
import torch
import torchaudio

from torchaudio.pipelines import SQUIM_OBJECTIVE
from torchinfo import summary
from tqdm import tqdm

from core.config import MAX_DURATION, SAMPLE_RATE


def parse_args():
    """Parse arguments for Ravnursson data preparation"""

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dataset",
        type=str,
        default="nort3160",
        help="Input dataset",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device on which a torch.Tensor is or will be allocated",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    dataset_folder = f"data/{args.input_dataset}"

    if torch.cuda.is_available() and "cuda" in args.device:
        device = args.device
    else:
        device = "cpu"

    # SQUIM does not require the corresponding clean speech as reference for speech assessment
    objective_model = SQUIM_OBJECTIVE.get_model().to(device)
    objective_model.eval()
    torch.compile(objective_model)
    summary(objective_model)

    exts = ("wav", "mp3")

    base_pattern = f"{dataset_folder}/**/audio"

    audio_files = []

    for ext in exts:
        audio_files.extend(glob(f"{base_pattern}/*/*.{ext}", recursive=True))

    sq_data = {}

    for file in tqdm(audio_files):
        with torch.no_grad():
            x, sr = torchaudio.load(file)

            if sr != SAMPLE_RATE:
                x = torchaudio.functional.resample(x, sr, SAMPLE_RATE)

            x = x[: int(MAX_DURATION * SAMPLE_RATE)].to(device)

            stoi, pesq, si_sdr = objective_model(x[0:1])

        sq_data[file] = {
            "stoi": stoi.cpu().item(),
            "pesq": pesq.cpu().item(),
            "si_sdr": si_sdr.cpu().item(),
        }

    sq_df = pd.DataFrame.from_dict(sq_data, orient="index")

    sq_df.to_csv(f"data/stats/speech_quality_{args.input_dataset}.csv")
