<!-- markdownlint-disable MD022 MD032 -->

# Llama Guard in Its Own Watcher on the GPU Host

## Status

Accepted

## Context

The realtime watcher rates every message with OpenAI moderation and Llama Guard
(see the [schema and watcher ADR](./2026-03-02-hcms-mongo-schema-alignment-and-realtime-monitoring.md)).
Llama Guard 4 (12B) is now self-hosted on a GPU machine on the institute
network. That machine can open outbound connections, including to MongoDB, but
nothing outside can reach it, so the watcher deployed in Kubernetes can't call
the model.

Two further problems showed up:

1. The Llama Guard rater sent every message alone as a user turn, so an
   assistant reply was judged as if a user had written it. For example, a reply
   explaining how to break into a neighbour's house was rated safe on its own,
   but unsafe (Non-Violent Crimes) as the reply to the user's question.
2. The watcher read `hcms_LOG_LEVEL`, while the docs and deployment set
   `HCMS_LOG_LEVEL`, so the log level setting was ignored.

## Decision

1. Add `HCMS_MODERATION_PROVIDERS` (comma-separated `openai_moderation`,
   `llama_guard`; default both). A watcher only rates, backfills and writes its
   own providers' `reviewer_flags` entries and ignores the others.
2. The Kubernetes watcher runs `openai_moderation` only; `LLAMA_GUARD_API_KEY`
   and `LLAMA_GUARD_ENDPOINT` are removed from its deployment.
3. A second watcher runs `llama_guard` only on the GPU host, one per database
   (dev and main), and talks to the model on localhost.
4. Llama Guard rates an assistant reply together with the closest earlier user
   message, as a user/assistant conversation. User messages, and replies with no
   user message before them, are still rated alone as a user turn.
5. Read `HCMS_LOG_LEVEL`.

## Consequences

- Both watchers write to the same messages without overwriting each other:
  each write is one atomic update that replaces or appends only the entry with
  its own `reviewer_id`.
- Existing `system_llama_guard` entries are kept; the GPU host's startup
  backfill only fills in messages that have none.
- Llama Guard ratings of assistant replies from the realtime watcher are not
  directly comparable with ratings made before this change. The offline dataset
  rater (`rate_conversations_with_llama_guard`) still rates each message alone.
- Context-aware ratings are cached under the whole conversation
  (`llama_guard_response`), so they never reuse a single-message result.
