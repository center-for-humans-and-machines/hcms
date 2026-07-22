<!-- markdownlint-disable MD022 MD032 -->

# Compose Restructure for Lint and Local Monitoring

## Status

Accepted

## Context

The repository used a single `compose.yaml` file that defined only the
`super-linter` service. This created two issues:

1. The default Compose filename implied it was the general project stack, but it
   was only for linting.
2. The new MongoDB watcher runtime for BAN-24 needs a dedicated local stack
   including MongoDB replica set initialization and the HCMS watcher process.

## Decision

Split Compose files by purpose:

1. `compose.lint.yml`
   - contains only the `super-linter` service.
2. `compose.yml`
   - contains local runtime services for monitoring:
     - `mongo`
     - `mongo-init-rs`
     - `watcher`

Update script wiring so linting is explicit:

- `script/lint-superlinter` now runs:
  - `docker compose -f compose.lint.yml run --rm --remove-orphans super-linter`

Add a dedicated watcher image build definition:

- `Dockerfile`
  - based on `python:3.13-slim`
  - installs Poetry and project dependencies
  - runs `python -m hcms.monitoring.watch_mongo_conversations`

Add explicit lifecycle scripts:

- `script/start`
  - runs `docker compose --file compose.yml up --detach --build --remove-orphans`
- `script/stop`
  - runs `docker compose --file compose.yml stop`
- `script/destroy`
  - runs `docker compose --file compose.yml down --volumes --remove-orphans`

## Local Monitoring Stack Architecture

### `mongo`

- image: `mongo:8.2`
- command: `--replSet rs0 --bind_ip_all`
- port: `27017:27017`
- persistent volume: `mongo_data`

### `mongo-init-rs`

- one-shot helper container
- waits until MongoDB is reachable
- initializes replica set if needed (`rs.initiate(...)`)
- exits cleanly if replica set is already initialized

### `watcher`

- built from `Dockerfile`
- depends on `mongo` and successful completion of `mongo-init-rs`
- runs with restart policy `unless-stopped`
- environment contract:
  - `MONGODB_URI=mongodb://mongo:27017/?replicaSet=rs0`
  - `MONGODB_DATABASE=hcms_local`
  - `MONGODB_COLLECTION=Conversations`
  - `MONGODB_CHANGE_STREAM_MAX_AWAIT_MS=1000`
  - `MONGODB_BACKFILL_BATCH_SIZE=200`
  - `MONGODB_BACKFILL_MAX_RETRIES=10`
  - `MONGODB_BACKFILL_RETRY_SLEEP_SECONDS=2`
  - `MONGODB_RECONNECT_BACKOFF_BASE_SECONDS=1`
  - `MONGODB_RECONNECT_BACKOFF_MAX_SECONDS=30`
  - `MONGODB_RECONNECT_BACKOFF_JITTER_SECONDS=0.25`
  - `HCMS_LOG_LEVEL=INFO`
  - `OPENAI_MODERATION_API_KEY`
  - `LLAMA_GUARD_API_KEY`
  - `LLAMA_GUARD_ENDPOINT`
- runtime behavior:
  - startup backfill retries transient provider failures with bounded retry budget
  - change stream reconnection uses exponential backoff with jitter
  - watcher logs periodic runtime counters and unresolved startup backfill targets

## Operational Commands

### Linting

- `./script/lint-superlinter`
- `docker compose -f compose.lint.yml run --rm --remove-orphans super-linter`

### Local Monitoring

- start:
  - `./script/start`
- stop:
  - `./script/stop`
- destroy (data loss):
  - `./script/destroy`

### Validation

Insert a valid conversation into `hcms_local.Conversations` and verify watcher
writes system `reviewer_flags` entries for:

- `system_openai_moderation`
- `system_llama_guard`

## Migration Impact

- `compose.yaml` is no longer the lint stack filename.
- `script/lint-superlinter` now has explicit compose file selection.
- local monitoring now has a canonical Compose entrypoint via
  `compose.yml`.
- lifecycle operations are explicit through start/stop/destroy scripts.

## Consequences

Positive:

- clearer separation between CI/local lint tooling and runtime services
- reproducible local monitoring stack for development and testing
- explicit script behavior independent of default Compose file naming

Trade-offs:

- additional Compose and Docker maintenance surface
- local monitoring requires Docker resources and provider credentials
