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

Paper dataset generation and evaluation instructions are documented in:

- [`docs/paper-data-workflow.md`](./docs/paper-data-workflow.md)

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
  ./script/start
  ```

- Stop stack (keep data):

  ```sh
  ./script/stop
  ```

- Destroy stack (remove containers + volumes):

  ```sh
  ./script/destroy
  ```

Watcher runtime behavior:

- Startup backfill retries transient provider failures (default `10` retries, `2s` between no-progress retries).
- Change stream reconnect uses exponential backoff with jitter (default `1s` base, `30s` max, `0.25s` jitter).
- Logging level defaults to `INFO` and can be tuned with `HPMS_LOG_LEVEL`.

Main tuning variables (set in `.env`):

- `MONGODB_CHANGE_STREAM_MAX_AWAIT_MS`
- `MONGODB_BACKFILL_BATCH_SIZE`
- `MONGODB_BACKFILL_MAX_RETRIES`
- `MONGODB_BACKFILL_RETRY_SLEEP_SECONDS`
- `MONGODB_RECONNECT_BACKOFF_BASE_SECONDS`
- `MONGODB_RECONNECT_BACKOFF_MAX_SECONDS`
- `MONGODB_RECONNECT_BACKOFF_JITTER_SECONDS`
- `HPMS_LOG_LEVEL`

Connect to local MongoDB from host tools (MongoDB Compass, mongosh, app clients):

```text
mongodb://localhost:27017/?directConnection=true
```

Note: the local replica set member is advertised as `mongo:27017` inside Docker.
From the host machine, use `directConnection=true` to avoid hostname resolution
errors for `mongo`.

## Documentation

- Architecture Decision Record (ADR) for compose split and local monitoring stack: [BAN-24 compose restructure](./docs/adr/2026-03-02-ban-24-compose-restructure-local.md)
- ADR for Mongo schema alignment and realtime watcher: [BAN-24 schema and watcher](./docs/adr/2026-03-02-ban-24-hpms-mongo-schema-alignment-and-realtime.md)
- Paper data workflow: [paper-data-workflow.md](./docs/paper-data-workflow.md)
- CI/CD deployment: [ci-cd-deployment.md](./docs/ci-cd-deployment.md)
- Contribution guide: [contributing.md](./contributing.md)

## Related

- Readme template: [minimal-readme](https://github.com/rodrigobdz/minimal-readme)
- Shell style: [styleguide-sh](https://github.com/rodrigobdz/styleguide-sh)

## License

[CC-BY-4.0](./license)
