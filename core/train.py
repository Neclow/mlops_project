"""End-to-end evaluation of embeddings for audio-based LID"""

import logging
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    LR,
    MAX_DURATION,
    N_EPOCHS,
    RANDOM_STATE,
    SAMPLE_RATE,
    WEIGHT_DECAY,
)
from .data import load_data
from .downstream import LightningMLP
from .transforms import load_transforms
from .upstream import load_feature_extractor, load_processor

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def parse_args():
    """Parse arguments for end-to-end LID evaluation"""

    default_root_dir = "/home/common/speech_phylo"

    parser = ArgumentParser(
        description="Arguments for end-to-end LID evaluation",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=str,
        default="nort3160",
        help="Dataset. Example: `nort3160`",
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="Pre-trained model",
    )
    parser.add_argument("dtype", choices=("text", "speech"), help="Data type")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_DURATION,
        help="Audio duration to feed in the dataset",
    )
    parser.add_argument(
        "--transform",
        action="store_true",
        help="Whether to transform the data using VAD",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device on which a torch.Tensor is or will be allocated",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="Weight decay",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=N_EPOCHS,
        help="Number of training epcohs",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to main data folder path",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=f"{default_root_dir}/models",
        help="Path where cached models are stored",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Random seed")

    return parser.parse_args()


def main():
    """Main loop"""
    args = parse_args()

    print("Configuration:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    seed_everything(args.seed, workers=True)

    print("Loading feature extractor...")
    feature_extractor = load_feature_extractor(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
    )

    print("Preparing data...")
    dataset_args = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
    }

    loader_args = {
        "num_workers": 4,
        "batch_size": args.batch_size,
        "pin_memory": True,
    }

    if args.dtype == "speech":
        print("Loading processor...")
        processor = load_processor(
            model_id=args.model_id,
            sr=SAMPLE_RATE,
            cache_dir=args.cache_dir,
        )

        transform = load_transforms(
            sr=SAMPLE_RATE, max_duration=args.max_duration, vad=args.transform
        )

        dataset_args = {
            **dataset_args,
            "processor": processor,
            "max_duration": args.max_duration,
            "transform": transform,
        }
    else:
        raise NotImplementedError

    print("Loading data...")
    train_dataset, valid_dataset, test_dataset = load_data(**dataset_args)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    print("Preparing downstream classifier...")
    lit_mlp = LightningMLP(
        feature_extractor=feature_extractor,
        num_classes=len(train_dataset.label_encoder),
        loss_fn=nn.NLLLoss(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    torch.compile(lit_mlp)

    # W & B monitoring and logging
    eval_dir = f"{args.data_dir}/eval"
    os.makedirs(eval_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project=f"mlops_project_eval_{args.dataset}", save_dir=eval_dir
    )

    wandb_logger.experiment.config.update(
        {
            "model_id": args.model_id,
            "max_duration": args.max_duration,
            "transform": args.transform,
        }
    )

    # TODO: make it more flexible for multi-GPU
    if "cuda" in args.device:
        accelerator = "gpu"
        devices = [int(args.device.split(":")[-1])]
    else:
        accelerator = "cpu"
        devices = "auto"

    print("Start training!")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.n_epochs,
        enable_model_summary=True,
        callbacks=[
            ModelCheckpoint(monitor="valid_loss", mode="min", save_last=True),
            TQDMProgressBar(),
        ],
        logger=wandb_logger,
    )

    trainer.fit(
        model=lit_mlp,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

    trainer.test(
        model=lit_mlp,
        dataloaders=test_loader,
        ckpt_path="best",
    )


if __name__ == "__main__":
    main()
