# MLOps project

## Dependencies

```bash
conda env create -f env.yml
conda activate phylaudio
pip install -r extra_requirements.txt
```

## Embedding evaluation

```bash
python -m core.eval fleurs-r whisper_tiny speech --n-epochs 3 --max-duration 10 --device cuda:1
```
