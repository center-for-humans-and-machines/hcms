# HPMS

Hybrid Psychiatrist-in-the-Loop Monitoring System (HPMS) for safety monitoring and evaluation of LLM conversational agents.

## Requirements

- Python `3.13+`
- [Poetry](https://python-poetry.org/) `2.1.2`
- [Docker](https://docs.docker.com/get-docker/) with Compose v2
- [`pre-commit`](https://pre-commit.com/)

## Installation

- Bootstrap local environment:

  ```sh
  ./script/bootstrap
  ```

- Install package dependencies:

  ```sh
  ./script/install
  ```

## Usage

### Paper

The data from the paper is available in the [`data/paper`](./data/paper/) folder and are needed to run some of the notebooks.
There are two main scripts to run the experiments:

- Generate dataset:

  ```sh
  poetry run python generate_dataset.py --round-number=2 --tag=acm-tist
  ```

- Evaluate dataset:

  ```sh
  poetry run python evaluate_dataset.py --round-number=2
  ```

Both scripts are executed in the [regression test pipeline](./.github/workflows/regression-test.yml) using GitHub Actions.

### Lint and tests

- Run linters:

  ```sh
  ./script/lint
  ```

- Run tests (excluding regression):

  ```sh
  ./script/test
  ```

- Run full tests (including regression):

  ```sh
  ./script/test -r
  ```

### Local monitoring stack

Use these lifecycle scripts for local MongoDB + watcher:

- Start stack:

  ```sh
  ./script/start-monitoring
  ```

- Stop stack (keep data):

  ```sh
  ./script/stop-monitoring
  ```

- Destroy stack (remove containers + volumes):

  ```sh
  ./script/destroy-monitoring
  ```

## Documentation

- ADR for compose split and local monitoring stack:
  - [BAN-24 compose restructure](./docs/adr/2026-03-02-ban-24-compose-restructure-local-monitoring.md)
- ADR for Mongo schema alignment and realtime watcher:
  - [BAN-24 schema and watcher](./docs/adr/2026-03-02-ban-24-hpms-mongo-schema-alignment-and-realtime-monitoring.md)
- Contribution guide:
  - [contributing.md](./contributing.md)

## Related

- Readme template:
  - [minimal-readme](https://github.com/rodrigobdz/minimal-readme)
- Shell style:
  - [styleguide-sh](https://github.com/rodrigobdz/styleguide-sh)

## License

[CC-BY-4.0](./license)
