# MLOps project

## Dependencies

```bash
conda env create -f env.yml
conda activate phylaudio
pip install -r extra_requirements.txt
```

## Embedding evaluation

```bash
python -m core.train fleurs-r NeMo_ambernet speech --n-epochs 3 --max-duration 10 --device cuda:1
```

## Data

### Languages

The .json file describes common abbreviations/codes for languages. For each language, the dictionary values always contain the full name (as described in Glottolog), as well as its glottocode ("glottolog" field).

These files were built semi-manually using information from the datasets, Glottolog, and Wikipedia.
