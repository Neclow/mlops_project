"""Ravnursson data preparation"""

import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pandas as pd
import soundfile as sf

from datasets import load_dataset
from tqdm import tqdm

from core.config import SAMPLE_RATE

MAX_SAMPLES = 10000


def parse_args():
    """Parse arguments for Ravnursson data preparation"""

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        default="nort3160",
        help="Output dataset folder",
    )
    parser.add_argument(
        "-sr", "--sr", default=SAMPLE_RATE, type=int, help="Sample rate"
    )
    parser.add_argument(
        "-n",
        "--max_samples",
        default=MAX_SAMPLES,
        type=int,
        help="Maximum number of samples to keep per split (train/dev/test)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Download dataset
    ravnursson = load_dataset("carlosdanielhernandezmena/ravnursson_asr")

    output_folder = f"data/{args.dataset_dir}/ravnursson/fo/fo"

    audio_folder = f"{output_folder}/audio"

    os.makedirs(audio_folder, exist_ok=True)

    for split in ["train", "validation", "test"]:
        # Create folder for subset
        if split == "validation":
            # Replace 'validation' with 'dev's
            split_folder = f"{audio_folder}/dev"
        else:
            split_folder = f"{audio_folder}/{split}"
        os.makedirs(split_folder, exist_ok=True)

        metadata = []

        for i in tqdm(range(min(len(ravnursson[split]), args.max_samples))):
            row = ravnursson[split][i]

            # name: filename
            # arr: audio loaded as a numpy array
            # sr: sample rate
            name, arr, sr = list(row["audio"].values())

            metadata.append(
                {
                    "name": name,
                    "transcript": row["normalized_text"],
                    "speaker_id": row["speaker_id"],
                    "gender": row["gender"],
                    "age": row["age"],
                    "dialect": row["dialect"],
                    "duration": row["duration"],
                }
            )

            # Re-sample audio and write as a wav
            sf.write(f"{split_folder}/{name}.wav", data=arr, samplerate=args.sr)

        meta_df = pd.DataFrame(metadata)

        meta_df.to_csv(f"{output_folder}/{split}.tsv", sep="\t")
