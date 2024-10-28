"""Common Voice 17.0 data preparation"""

import os
import subprocess

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob
from pathlib import Path

import requests

from tqdm import tqdm

# Languages
langs = [
    "da",
    "is",
    "nn-NO",
    "sv-SE",
]

# Max time before timeout
TIMEOUT = 10
# Root URL
ROOT_URL = (
    "https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0/resolve/main"
)


def parse_args():
    """Parse arguments for Common Voice data preparation"""

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Output dataset folder",
    )
    return parser.parse_args()


def untar(dataset_dir):
    """Example code to decompress all the downloaded tar files"""
    tar_files = glob(f"data/{dataset_dir}/common_voice_17_0/*/*/audio/*/*.tar")
    for tf in tqdm(tar_files):
        tf_parent = str(Path(tf).parent)
        commands = (
            f"cd {tf_parent} && tar -xf {os.path.basename(tf)} --strip-components=1"
        )
        subprocess.run(commands, check=True, shell=True)


def download(dataset_dir, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"}

    for lang in langs:
        print(lang)
        lang_folder = f"data/{dataset_dir}/common_voice_17_0/{lang}/{lang}"
        audio_folder = f"{lang_folder}/audio"
        os.makedirs(lang_folder, exist_ok=True)
        os.makedirs(audio_folder, exist_ok=True)
        for split in ["train", "dev", "test"]:
            # Download transcripts
            transcript_url = f"{ROOT_URL}/transcript/{lang}/{split}.tsv"
            response = requests.get(transcript_url, headers=headers, timeout=TIMEOUT)

            if response.status_code == 200:
                print(f"Downloading {transcript_url}")

                with open(
                    f"{lang_folder}/{os.path.basename(transcript_url)}", "wb"
                ) as f:
                    f.write(response.content)
            else:
                continue

            # Download audio
            split_folder = f"{audio_folder}/{split}"
            os.makedirs(split_folder, exist_ok=True)

            i = 0

            while True:
                audio_url = f"{ROOT_URL}/audio/{lang}/{split}/{lang}_{split}_{i}.tar"
                response = requests.get(audio_url, headers=headers, timeout=TIMEOUT)

                if response.status_code == 200:
                    print(f"Downloading {audio_url}")
                    with open(
                        f"{split_folder}/{os.path.basename(audio_url)}", "wb"
                    ) as f:
                        f.write(response.content)
                    i += 1
                else:
                    break


def main():
    args = parse_args()

    hf_token = input("Enter your HF token here: ")

    download(dataset_dir=args.dataset_dir, hf_token=hf_token)

    untar(dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
