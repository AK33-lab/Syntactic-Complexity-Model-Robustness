# Syntactic Complexity & Model Robustness

This project evaluates four NLI models on original and perturbed MNLI premises and writes two CSV outputs:

- `perf.csv`: `model`, `perturbation method`, `performance`
- `complex.csv`: `perturbation method`, `metric type`, `value`

## Main Script

Run from the root folder:

```bash
python main.py
```

The script evaluates these models:

- MLP
- RNN
- RoBERTa
- BART

Across these methods:

- original
- adjrel
- passive
- pp

## Required Files

- `main.py`
- `models.py`
- `evaluate.py`
- `data_loader.py`
- `metrics.py`
- `perturbations.py`
- `data.jsonl`
- `mlp_weights.pt`
- `rnn_weights.pt`

## Environment

Install packages:

```bash
pip install torch transformers "datasets" spacy benepar pyinflect tqdm numpy
```

Download language/parser models:

```bash
python -m spacy download en_core_web_sm
python -m benepar.download benepar_en3
```

## Notes

- If a CUDA GPU is available, the code uses it automatically.
- Running `main.py` rewrites `perf.csv` and `complex.csv`.
