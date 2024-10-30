"""Constants"""

from typing import Final

BATCH_SIZE: Final = 64
# Learning rate
LR: Final = 2.5e-4
# Number of training epochs
N_EPOCHS: Final = 3
# Audio duration to feed in the dataset
MAX_DURATION: Final = 30.0
RANDOM_STATE: Final = 42
SAMPLE_RATE: Final = 16000
# Weight decay
WEIGHT_DECAY: Final = 1e-3

MODEL_IDS: Final = [
    "facebook/wav2vec2-xls-r-300m",
    "facebook/mms-lid-126",
    "NeMo_ambernet",
    "speechbrain/lang-id-voxlingua107-ecapa",
    "whisper_tiny",
    "whisper_base",
]
