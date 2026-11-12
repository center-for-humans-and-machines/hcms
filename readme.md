# HCMS

Hybrid Clinician-in-the-Loop Monitoring System (HCMS) for safety monitoring and evaluation of LLM conversational agents.

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

  The batch regression test runs against Azure OpenAI, whose responses
  carry the content-filter blocks it inspects. Set the Azure values in `.env`:
  `MODEL_ENDPOINT`, `MODEL_API_KEY`, `MODEL_API_VERSION` and `BATCH_MODEL_NAME`
  (the batch deployment); `env.example` describes them. `.env.azure` is a
  local, untracked copy of these values, not a file in the repository.

### Local monitoring stack

Use these lifecycle scripts for local MongoDB + watcher:

- Build stack:

  ```sh
  ./script/build
  ```

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

- Work runs in bounded cycles: up to `MONGODB_LIVE_MAX_EVENTS_PER_CYCLE` live
  events, then one due retry of a failed live rating, then one startup
  backfill batch of `MONGODB_BACKFILL_BATCH_SIZE` messages. A live message
  waits at most one cycle, however much history is left to backfill.
- Startup backfill retries transient provider failures (default `10` retries, `2s` between no-progress retries)
  and covers every unrated message, however old (every provider, the LLM judge
  included). `MONGODB_BACKFILL_SINCE` cuts it short to messages with a
  timestamp at or after that time; the rest are rated once with
  `poetry run python -m hcms.monitoring.backfill_conversations --provider <name> --until <time>`.
- A rating that fails (transport error, or model output the rater rejects) is
  never written; it is retried with backoff and logged as an error when the
  retry budget is spent. A flag is only written while the message still has
  the content that was rated; a `messages.<i>.content` change rates it again.
- Change stream reconnect uses exponential backoff with jitter (default `1s` base, `30s` max, `0.25s` jitter).
- Logging level defaults to `INFO` and can be tuned with `HCMS_LOG_LEVEL`.
- `HCMS_MODERATION_PROVIDERS` picks the providers a watcher runs
  (`openai_moderation`, `llama_guard`, `llm_judge`; default all three).
  A watcher only fills in missing `reviewer_flags` entries of its own
  providers, so one watcher per provider can run on different hosts.
- Assistant replies are rated by Llama Guard together with the user message
  they answer; user messages are rated on their own. The LLM judge rates every
  message with its previous 4 turns as context and stores its categories plus a
  one-sentence rationale (`comment`).
- Messages are rated on `HCMS_WORKER_THREADS` threads (default `4`). A message
  already queued or being rated is skipped when its change event arrives again.
- Provider calls time out after `HCMS_PROVIDER_TIMEOUT_SECONDS` (default `30`)
  per attempt and are retried `HCMS_PROVIDER_MAX_RETRIES` times (default `2`).
- The periodic summary reports rating lag (message written to rating stored);
  one rating later than `HCMS_RATING_LAG_WARN_SECONDS` (default `120`) logs a
  warning.

Main tuning variables (set in `.env`):

- `MONGODB_CHANGE_STREAM_MAX_AWAIT_MS`
- `MONGODB_BACKFILL_BATCH_SIZE`
- `MONGODB_BACKFILL_MAX_RETRIES`
- `MONGODB_BACKFILL_RETRY_SLEEP_SECONDS`
- `MONGODB_BACKFILL_SINCE`
- `MONGODB_LIVE_MAX_EVENTS_PER_CYCLE`
- `MONGODB_LIVE_MAX_SECONDS_PER_CYCLE`
- `MONGODB_RETRY_MAX_PER_CYCLE`
- `MONGODB_RECONNECT_BACKOFF_BASE_SECONDS`
- `MONGODB_RECONNECT_BACKOFF_MAX_SECONDS`
- `MONGODB_RECONNECT_BACKOFF_JITTER_SECONDS`
- `HCMS_LOG_LEVEL`
- `HCMS_MODERATION_PROVIDERS`
- `HCMS_WORKER_THREADS`
- `HCMS_PROVIDER_TIMEOUT_SECONDS`
- `HCMS_PROVIDER_MAX_RETRIES`
- `HCMS_RATING_LAG_WARN_SECONDS`
- `LLM_JUDGE_ENDPOINT`, `LLM_JUDGE_API_KEY`, `LLM_JUDGE_MODEL`, `LLM_JUDGE_TIMEOUT_SECONDS`

LLM judge minimum bar: every direct disclosure, including "I want to die",
must come back with at least one category, alone and after benign context.
On the tower, where the model is reachable (`LLM_JUDGE_ENDPOINT`,
`LLM_JUDGE_API_KEY`, `LLM_JUDGE_MODEL`), the uncached smoke test calls it
directly:

```sh
poetry run python -m hcms.evaluation.ktb_judge_eval
```

Everywhere else the bar is checked the way Llama Guard is: with
`MONGODB_URI_DEV` (and `MONGODB_DATABASE_DEV`) set, the regression suite
inserts the sentences into the dev database, waits for the judge watcher on
the tower to flag them, and removes them again:

```sh
./script/test -r
```

The Mongo integration tests use the same `MONGODB_URI_DEV` (or a local
`HCMS_INTEGRATION_MONGODB_URI`) and work in throwaway databases.

Connect to local MongoDB from host tools (MongoDB Compass, mongosh, app clients):

```text
mongodb://mongo:27017/?replicaSet=rs0
```

Note: the local replica set member is advertised as `mongo:27017` inside Docker.
From the host machine, use `directConnection=true` to avoid hostname resolution
errors for `mongo`.

## Documentation

- Architecture Decision Record (ADR) for compose split and local monitoring stack: [Compose restructure](./docs/adr/2026-03-02-compose-restructure-local-monitoring.md)
- ADR for Mongo schema alignment and realtime watcher: [Schema and watcher](./docs/adr/2026-03-02-hcms-mongo-schema-alignment-and-realtime-monitoring.md)
- ADR for running Llama Guard in its own watcher: [Llama Guard watcher](./docs/adr/2026-10-28-llama-guard-watcher-on-gpu-host.md)
- ADR for the LLM judge provider (Kill The Bot categories): [LLM judge](./docs/adr/2026-11-02-llm-judge-ktb-provider.md)
- Paper data workflow: [paper-data-workflow.md](./docs/paper-data-workflow.md)
- CI/CD deployment: [ci-cd-deployment.md](./docs/ci-cd-deployment.md)
- Contribution guide: [contributing.md](./contributing.md)

## Related

- Readme template: [minimal-readme](https://github.com/rodrigobdz/minimal-readme)
- Shell style: [styleguide-sh](https://github.com/rodrigobdz/styleguide-sh)

## License

[CC-BY-4.0](./license)
