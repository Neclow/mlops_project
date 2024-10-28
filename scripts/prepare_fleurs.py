"""FLEURS-R data preparation"""

import os
import subprocess
import warnings

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob
from pathlib import Path

import requests

from tqdm import tqdm

# Languages
langs = ["da_dk", "is_is", "nb_no", "sv_se"]
# Max time before timeout
TIMEOUT = 10
# Root URL
ROOT_URL = "https://huggingface.co/datasets/google/fleurs-r/resolve/main/data"


def parse_args():
    """Parse arguments for FLEURS-R data preparation"""

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Output dataset folder",
    )
    return parser.parse_args()


def unzip(dataset_dir):
    """Example code to decompress all the downloaded tar.gz files"""
    tar_files = glob(f"data/{dataset_dir}/fleurs-r/*/*/audio/*/*.tar.gz")
    for tf in tqdm(tar_files):
        tf_parent = str(Path(tf).parent)
        commands = (
            f"cd {tf_parent} && tar -xzf {os.path.basename(tf)} --strip-components=1"
        )
        subprocess.run(commands, check=True, shell=True)


def download(dataset_dir):
    for lang in langs:
        print(lang)
        lang_folder = f"data/{dataset_dir}/fleurs-r/{lang}/{lang}"
        audio_folder = f"{lang_folder}/audio"
        os.makedirs(lang_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)
        for split in ["train", "dev", "test"]:
            # Donwload transcripts
            transcript_url = f"{ROOT_URL}/{lang}/{split}.tsv"
            response = requests.get(transcript_url, timeout=TIMEOUT)

            if response.status_code == 200:
                print(f"Downloading {transcript_url}")

                with open(
                    f"{lang_folder}/{os.path.basename(transcript_url)}", "wb"
                ) as ft:
                    ft.write(response.content)
            else:
                warnings.warn(f"File not found: {transcript_url}", UserWarning)
                continue

            # Download audio
            split_folder = f"{audio_folder}/{split}"
            os.makedirs(split_folder, exist_ok=True)

            audio_url = f"{ROOT_URL}/{lang}/audio/{split}.tar.gz"

            response = requests.get(audio_url, timeout=TIMEOUT)

            if response.status_code == 200:
                print(f"Downloading {audio_url}")
                with open(f"{split_folder}/{os.path.basename(audio_url)}", "wb") as fa:
                    fa.write(response.content)


def main():
    args = parse_args()

    download(dataset_dir=args.dataset_dir)

    unzip(dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
