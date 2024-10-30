import json

import torch
import wandb

from ..config import SAMPLE_RATE
from ..downstream import LightningMLP
from ..upstream import load_feature_extractor, load_processor


def load_model_from_run_id(
    run_id, save_dir, project, cache_dir=None, device="cpu", **kwargs
):
    # save_dir = "../data/eval"
    # project = "mlops_project_eval_nort3160"

    api = wandb.Api()

    run = api.run(f"neclow/mlops_project_eval_nort3160/{run_id}")

    ckpt = torch.load(f"{save_dir}/{project}/{run_id}/checkpoints/last.ckpt")

    state_dict = ckpt["state_dict"]

    processor = load_processor(
        run.config["model_id"], sr=SAMPLE_RATE, cache_dir=cache_dir
    )

    feature_extractor = load_feature_extractor(
        run.config["model_id"], cache_dir=cache_dir, device=device, **kwargs
    )

    lit_mlp = LightningMLP(
        feature_extractor=feature_extractor,
        num_classes=ckpt["hyper_parameters"]["num_classes"],
        loss_fn=ckpt["hyper_parameters"]["loss_fn"],
        lr=ckpt["hyper_parameters"]["lr"],
        weight_decay=ckpt["hyper_parameters"]["weight_decay"],
    )

    # We have restored our model
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
