# Stage 1: Build
FROM python:3.14-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_HOME=/opt/poetry
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# hadolint ignore=DL3008
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
  && poetry install --no-root --no-interaction --no-ansi

# Stage 2: Runtime
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# UID/GID above 10000 avoid clashes with host users; keep in sync with
# security.run_as_user/run_as_group in kubernetes/worker-service/values.yaml
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --gid "${APP_GID}" app \
  && useradd --uid "${APP_UID}" --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

# Keep the Python version in this path in sync with the FROM tags above
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages

# Source stays root-owned (read-only for the app); only the moderation
# cache directory (hcms/utils/cache.py) needs to be writable
COPY . .
RUN mkdir -p data/dataset/cache \
  && chown app:app data/dataset/cache

USER ${APP_UID}:${APP_GID}

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import hcms.database.models, pymongo, pydantic"]

CMD ["python", "-m", "hcms.monitoring.watch_mongo_conversations"]
