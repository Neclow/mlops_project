# MLOps project

## Dependencies

```bash
conda create --name audio python=3.10.10
pip install -r requirements.txt
pip install fairseq --no-deps
pip install bitarray sacrebleu omegaconf hydra-core
```

## Project components

### Data quality assurance

Uses Great Expectations + custom scripts.

Follow the ```notebooks/ata_quality.ipynb``` notebook.

Great Expectations outputs are available at ```mlops/great_expectations```.

To reproduce the speech quality assessment estimates, use:

```bash
python -m scripts.squim nort3160
```

### Model development

Uses Weights & Biases.

To train a classifier based on a pre-trained feature extractor:

```bash
python -m core.train nort3160 NeMo_ambernet speech --n-epochs 3 --max-duration 10 --device cuda:1
```

Or follow the ```notebooks/model_development.ipynb``` notebook.

A checkpoint for Titanet-LID is available at: <https://api.wandb.ai/artifactsV2/default/neclow/QXJ0aWZhY3Q6MTMwMDg1NDU0MQ%3D%3D/5f77a00d041aae409ae84c6cd334c011/last.ckpt>

### Model deployment

In a run, go to the root of the project, and run:

```bash
uvicorn mlops.fastapi.deploy:app --reload
```

Source code is available at ```mlops/fastapi/deploy.py```

Or follow the ```notebooks/model_deployment.ipynb``` notebook.

### Monitoring and logging

Follow the ```notebooks/model_monitoring.ipynb``` notebook.

## Data

### Download

Run the ```prepare``` scripts in ```scripts```:

```bash
python -m script.prepare_common_voice_17_0 nort3160
python -m script.prepare_fleurs nort3160
python -m script.prepare_ravnursson nort3160
```

### Languages

The .json file describes common abbreviations/codes for languages. For each language, the dictionary values always contain the full name (as described in Glottolog), as well as its glottocode ("glottolog" field).

These files were built semi-manually using information from the datasets, Glottolog, and Wikipedia.

### Stats

Statistics on the used datasets: sizes, durations, and speech quality

## Reports

MLOps_Project.pdf: assignment report
MLOps_Project.pptx: prototype slides
