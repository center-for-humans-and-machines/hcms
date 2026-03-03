# Paper Data Workflow

This page documents the dataset generation and evaluation workflow used for paper experiments.

## Paper Datasets

The data from the paper is available in [`data/paper`](../data/paper/) and is needed to run some notebooks.

The two main experiment scripts are:

- `generate_dataset.py`: generates synthetic conversational datasets
- `evaluate_dataset.py`: evaluates datasets using automated metrics

Both scripts are executed in the [regression test pipeline](../.github/workflows/regression-test.yml) with GitHub Actions.

## Data Generation

1. Review `.env` to set provider and model.
2. Repeat generation for each model.

Round 2:

```sh
poetry run python generate_dataset.py --round-number=2 --tag=acm-tist
```

Round 3:

```sh
poetry run python generate_dataset.py --round-number=3 --tag=acm-tist
```

## Data Evaluation

1. Repeat evaluation for each model.

Round 2:

```sh
poetry run python evaluate_dataset.py --round-number=2
```

Round 3:

```sh
poetry run python evaluate_dataset.py --round-number=3
```
