import os

import pandas as pd
import soundfile as sf

from datasets import load_dataset
from tqdm import tqdm

MAX_SAMPLES = 10000
SAMPLE_RATE = 16000

if __name__ == "__main__":
    ravnursson = load_dataset("carlosdanielhernandezmena/ravnursson_asr")

    dataset_folder = "data/datasets/ravnursson/fo/fo"

    audio_folder = f"{dataset_folder}/audio"

    os.makedirs(audio_folder, exist_ok=True)

    for split in ["train", "validation", "test"]:
        if split == "validation":
            split_folder = f"{audio_folder}/dev"
        else:
            split_folder = f"{audio_folder}/{split}"
        os.makedirs(split_folder, exist_ok=True)

        metadata = []

        for i in tqdm(range(min(len(ravnursson[split]), MAX_SAMPLES))):
            row = ravnursson[split][i]
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

            sf.write(f"{split_folder}/{name}.wav", data=arr, samplerate=SAMPLE_RATE)

        meta_df = pd.DataFrame(metadata)

        meta_df.to_csv(f"{dataset_folder}/{split}.tsv", sep="\t")
