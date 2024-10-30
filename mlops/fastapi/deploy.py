import io
import os

import pandas as pd
import torch

from fastapi import FastAPI, File, UploadFile
from speechbrain.dataio.encoder import CategoricalEncoder

from core.utils.model import get_run_ids_for_model_id, load_model_from_run_id

model_id = "NeMo_ambernet"
dataset = "nort3160"
data_dir = "data"

# Find run ID
print("Getting runs...")
entity = "neclow"
save_dir = f"{data_dir}/eval"
project = "mlops_project_eval_nort3160"
run_ids = get_run_ids_for_model_id(model_id, entity=entity, project=project)

print(f"Found {len(run_ids)} with model_id: {model_id}. Selecting the first run.")

assert len(run_ids) > 0

my_cache_dir = "/home/common/speech_phylo/models"
if not os.path.isdir(my_cache_dir):
    my_cache_dir = None  # Or enter your cache dir here

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

print("Loading processor and model...")
processor, model = load_model_from_run_id(
    run_ids[0],
    save_dir=save_dir,
    project=project,
    cache_dir=my_cache_dir,
    device=device,
)
model.eval()
model.to(device)

print("Loading label encoder...")
language_metadata = pd.read_csv(f"{data_dir}/languages/{dataset}.csv")
languages = language_metadata.language.to_list()
label_dir = f"{data_dir}/labels"
label_encoder = CategoricalEncoder()
label_encoder.load_or_create(
    path=os.path.join(label_dir, f"{dataset}.txt"),
    from_iterables=[languages],
    output_key="lang_id",
)

# Start API
app = FastAPI(
    title="MLops project", description=f"Deployment of {model_id} trained on {dataset}"
)


@app.get("/index")
async def hello_world():
    return "hello world"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load audio file and convert to tensor
    audio_bytes = await file.read()

    x = processor(io.BytesIO(audio_bytes)).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        # Convert logits to prob
        probs = model.logits(x).exp()

    pred_prob, pred_label = probs.cpu().max(1)

    return {
        "label": label_encoder.decode_ndim(pred_label.item()),
        "prob": pred_prob.item(),
    }
