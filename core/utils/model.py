import json

import torch
import wandb

from ..config import SAMPLE_RATE
from ..downstream import LightningMLP
from ..upstream import load_feature_extractor, load_processor


def load_model_from_run_id(
    run_id, save_dir, entity, project, cache_dir=None, device="cpu", **kwargs
):
    # save_dir = "../data/eval"
    # project = "mlops_project_eval_nort3160"

    api = wandb.Api()

    run = api.run(f"{entity}/{project}/{run_id}")
    model_id = run.config["model_id"]

    # Load processor and feature extractor (without run weights)
    processor = load_processor(model_id, sr=SAMPLE_RATE, cache_dir=cache_dir)

    feature_extractor = load_feature_extractor(
        model_id, cache_dir=cache_dir, device=device, **kwargs
    )

    # Load checkpoint
    ckpt = torch.load(
        f"{save_dir}/{project}/{run_id}/checkpoints/last.ckpt", map_location=device
    )

    # Load Lightning Module
    hparams = ckpt["hyper_parameters"]
    lit_mlp = LightningMLP(
        feature_extractor=feature_extractor,
        num_classes=hparams["num_classes"],
        loss_fn=hparams["loss_fn"],
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )

    # Restore the model
    state_dict = ckpt["state_dict"]
    lit_mlp.load_state_dict(state_dict)

    return processor, lit_mlp


def get_run_ids_for_model_id(model_id, entity, project):
    # entity: neclow
    # project: mlops_project_eval_nort3160"

    run_ids = []

    api = wandb.Api()

    runs = api.runs(f"{entity}/{project}")

    for run in runs:
        run_config = json.loads(run.json_config)
        if run_config["model_id"]["value"] == model_id:
            run_ids.append(run.id)

    return run_ids
