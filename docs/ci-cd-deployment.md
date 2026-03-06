# CI/CD Deployment (GitHub Actions + Helm)

This repository deploys the monitoring `watcher` service through GitHub Actions by building a Docker image and deploying it to Kubernetes with Helm.

## Workflow Architecture

- Workflow: [`../.github/workflows/deployment.yml`](../.github/workflows/deployment.yml)
- Composite action (vars): [`../.github/actions/resolve-service-vars/action.yml`](../.github/actions/resolve-service-vars/action.yml)
- Composite action (deploy): [`../.github/actions/deploy-helm/action.yml`](../.github/actions/deploy-helm/action.yml)
- Helm chart: [`../kubernetes/worker-service`](../kubernetes/worker-service)

The workflow runs build and deploy pipelines for `watcher` and then writes a deployment summary.

## Branches and Triggers

Supported deployment branches:

- `dev`
- `main`

Workflow triggers:

- push to `dev` or `main`
- manual dispatch (`workflow_dispatch`)

## Runner and Environment Model

- All jobs run on `self-hosted` runners.
- Global workflow environment:
  - `REGISTRY`: `${{ vars.GITLAB_REGISTRY }}`
  - `DOCKER_REGISTRY_PREFIX`: `${{ vars.DOCKER_REGISTRY_PREFIX || 'mpib/chm/common/base-images' }}`
  - `NAMESPACE`: `${{ vars.K8S_NAMESPACE || 'elderbot' }}`

## Build and Deploy Logic

The `watcher` service follows this flow:

1. Resolve deterministic app/image naming (`<APP_NAME>-watcher-<branch>`).
1. Build and push Docker image from `Dockerfile.monitoring`.
1. Generate `/tmp/deployment_vars.yml` for runtime environment values.
1. Deploy with local `deploy-helm` composite action.

### Runtime deployment variables

The workflow writes this contract into `/tmp/deployment_vars.yml`:

Required values:

- `MONGODB_URI` (branch-specific: `MONGODB_URI_DEV`/`MONGODB_URI_MAIN`)
- `MONGODB_DATABASE` (branch-specific: `MONGODB_DATABASE_DEV`/`MONGODB_DATABASE_MAIN`)
- `OPENAI_MODERATION_API_KEY`
- `LLAMA_GUARD_API_KEY`
- `LLAMA_GUARD_ENDPOINT`

Optional values with defaults:

- `MONGODB_COLLECTION` (`Conversations`)
- `MONGODB_CHANGE_STREAM_MAX_AWAIT_MS` (`1000`)
- `MONGODB_BACKFILL_BATCH_SIZE` (`200`)
- `MONGODB_BACKFILL_MAX_RETRIES` (`10`)
- `MONGODB_BACKFILL_RETRY_SLEEP_SECONDS` (`2`)
- `MONGODB_RECONNECT_BACKOFF_BASE_SECONDS` (`1`)
- `MONGODB_RECONNECT_BACKOFF_MAX_SECONDS` (`30`)
- `MONGODB_RECONNECT_BACKOFF_JITTER_SECONDS` (`0.25`)
- `HPMS_LOG_LEVEL` (`INFO`)

## Required GitHub Variables

| Variable | Purpose |
|----------|---------|
| `GITLAB_REGISTRY` | Docker registry host |
| `APP_NAME` | Application name prefix used in image/release naming |
| `LLAMA_GUARD_ENDPOINT` | Llama Guard endpoint URL |
| `MONGODB_DATABASE_DEV` | MongoDB database name for `dev` deployments |
| `MONGODB_DATABASE_MAIN` | MongoDB database name for `main` deployments |

Optional variables:

- `DOCKER_REGISTRY_PREFIX` (default `mpib/chm/common/base-images`)
- `K8S_NAMESPACE` (default `elderbot`)
- `WATCHER_REPLICA_COUNT` (default `1`)
- watcher tuning variables listed above

## Required GitHub Secrets

| Secret | Purpose |
|--------|---------|
| `GITLAB_REGISTRY_USERNAME` | Registry authentication |
| `GITLAB_REGISTRY_PASSWORD` | Registry authentication |
| `KUBECONFIG` | Base64-encoded kubeconfig used for Helm deployment |
| `DOCKERCFG` | Base64 Docker config JSON for Kubernetes image pull secret |
| `MONGODB_URI_DEV` | Watcher MongoDB URI for `dev` deployments |
| `MONGODB_URI_MAIN` | Watcher MongoDB URI for `main` deployments |
| `OPENAI_MODERATION_API_KEY` | OpenAI moderation key used by watcher |
| `LLAMA_GUARD_API_KEY` | Llama Guard API key used by watcher |

## Kubernetes Chart Contract

The chart in `kubernetes/worker-service` defines a worker deployment contract with:

- image pull `Secret` (`kubernetes.io/dockerconfigjson`)
- `Deployment`

The worker chart intentionally does not create `Service` or `Ingress` resources because the watcher is a background consumer and not an HTTP service.

Required chart values provided by workflow/action:

- `app_name`
- `app_image`
- `dockersecret`
- `replica_count`

Runtime env values are passed through `deployment_vars` from `/tmp/deployment_vars.yml`.

## Deployment Summary

The final `deployment-summary` job always runs and writes one watcher row with:

- build status
- deploy status
- branch
- image tag and full image reference

## Troubleshooting

- Missing chart files: check `kubernetes/worker-service` exists in the repository.
- Registry auth failures: verify `GITLAB_REGISTRY_USERNAME` and `GITLAB_REGISTRY_PASSWORD`.
- Kubernetes auth failures: verify `KUBECONFIG` is valid base64 kubeconfig content.
- Missing watcher env values: verify required variables and secrets are configured in GitHub repository settings.
