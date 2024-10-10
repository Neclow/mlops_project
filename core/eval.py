"""End-to-end evaluation of embeddings for audio- or text-based LID"""

import logging
import os

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from speechbrain.dataio.batch import PaddedBatch
from torch import nn
from torch.utils.data import DataLoader

from core import downstream, upstream
from core.data import load_data
from core.transforms import Trim

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
        help="Dataset. Example: `fleurs`",
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="Pre-trained model",
    )
    parser.add_argument("dtype", choices=("text", "speech"), help="Data type")
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30,
        help="Audio duration to feed in the dataset",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Whether to load a finetuned XLS-R model",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device on which a torch.Tensor is or will be allocated",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2.5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=3,
        help="Number of training epcohs",
    )
    parser.add_argument(
        "--root-dir",
        default=default_root_dir,
        help="Root folder path for analyses",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=f"{default_root_dir}/models",
        help="Path where cached models are stored",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data",
        help="Path to save artefacts created during the eval",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    """Main loop"""
    args = parse_args()

    print("Configuration:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    seed_everything(args.seed, workers=True)

    print("Loading feature extractor...")
    feature_extractor = upstream.load_feature_extractor(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        finetuned=args.finetuned,
        device=args.device,
    )

    print("Preparing data...")
    dataset_args = {
        "dataset": args.dataset,
        "root_dir": args.root_dir,
        "save_dir": f"{args.save_dir}/labels",
    }

    loader_args = {
        "num_workers": 4,
        "batch_size": args.batch_size,
        "pin_memory": True,
    }

    if args.dtype == "speech":
        print("Loading processor...")
        processor = upstream.load_processor(
            model_id=args.model_id,
            sr=args.sr,
            cache_dir=args.cache_dir,
        )

        dataset_args = {
            **dataset_args,
            "processor": processor,
            # "is_whisper": "whisper" in args.model_id,
            "max_duration": args.max_duration,
            "transform": Trim(sample_rate=args.sr, max_duration=args.max_duration),
        }

        loader_args["collate_fn"] = PaddedBatch

    train_dataset, valid_dataset, test_dataset = load_data(**dataset_args)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)

    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_args)

    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    print("Preparing downstream classifier...")
    lit_mlp = downstream.LightningMLP(
        feature_extractor=feature_extractor,
        num_classes=len(train_dataset.label_encoder),
        loss_fn=nn.NLLLoss(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    torch.compile(lit_mlp)

    # ModelSummary(lit_mlp, max_depth=4)

    eval_dir = f"{args.save_dir}/eval"
    os.makedirs(eval_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        project=f"phylaudio_eval_{args.dataset}", save_dir=eval_dir
    )

    wandb_logger.experiment.config.update(
        {
            "model_id": args.model_id,
            "finetuned": args.finetuned,
            "max_duration": args.max_duration,
        }
    )

    # TODO: make it more flexible for multi-GPU
    if "cuda" in args.device:
        accelerator = "gpu"
        devices = [int(args.device.split(":")[-1])]
    else:
        accelerator = "cpu"
        devices = "auto"

    print("Start evaluation!")
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=args.n_epochs,
        # precision="16-mixed",
        enable_model_summary=True,
        callbacks=[
            ModelCheckpoint(monitor="valid_loss", mode="min", save_last=True),
            TQDMProgressBar(),
        ],
        logger=wandb_logger,
        # default_root_dir=f"{args.save_dir}/eval",
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
