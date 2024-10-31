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

Follow the ```data_quality.ipynb``` notebook.

### Model development

Uses Weights & Biases.

To train a classifier based on a pre-trained feature extractor:

```bash
python -m core.train nort3160 NeMo_ambernet speech --n-epochs 3 --max-duration 10 --device cuda:1
```

Or follow the ```model_development.ipynb``` notebook.

### Model deployment

TODO (FastAPI)

### Monitoring and logging

TODO (Evidently)

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
