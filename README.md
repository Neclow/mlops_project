# MLOps project

## Dependencies

```bash
conda env create -f env.yml
conda activate phylaudio
pip install -r extra_requirements.txt
```

## Project components

### Data quality assurance

Follow the ```data_quality.ipynb``` notebook.

### Model development

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
python -m prepare_common_voice_17_0 nort3160
python -m prepare_fleurs nort3160
python -m prepare_ravnursson nort3160
```

### Languages

The .json file describes common abbreviations/codes for languages. For each language, the dictionary values always contain the full name (as described in Glottolog), as well as its glottocode ("glottolog" field).

These files were built semi-manually using information from the datasets, Glottolog, and Wikipedia.

### Stats

Statistics on the used datasets: sizes, durations, and speech quality
