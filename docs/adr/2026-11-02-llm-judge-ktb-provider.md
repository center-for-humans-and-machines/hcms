<!-- markdownlint-disable MD022 MD032 -->

# LLM Judge Provider With Kill The Bot Categories

## Status

Accepted (BAN-1323)

## Context

Llama Guard 4, self-hosted on the RCT tower (see the
[Llama Guard watcher ADR](./2026-10-28-llama-guard-watcher-on-gpu-host.md)), is
the only on-site classifier rating Elderbot RCT messages, and it misses plain
suicidal ideation: "I want to die" comes back `safe`. A second, independent
classifier is needed so such a message is caught by at least one of the two.

The tower's GPU is full (Llama Guard F16 uses 6.4 of 8 GiB), it has 100 GiB of
free RAM and 24 CPU threads, and nothing outside can reach it. The Kill The Bot
(KTB) scale from the psychiatrists' red-teaming study gives the harm domains the
clinical team cares about: suicidality (methods, reasons, "positive" aspects,
cover stories), self-harm and substance use. KTB has no item for a spontaneous
wish to die, and its 0 to −4 ratings grade the bot's reply, not the user's
message.

Three constraints shaped the watcher side:

1. The watcher used to finish the whole startup backfill before reading live
   events, and a failed live rating was only retried by the next restart. A
   judge that needs 10–15 s per message on the CPU would have stalled live
   detection for days.
2. A cutoff by conversation `created_at` would leave messages added later to
   older conversations unrated forever.
3. The dashboard treats a non-empty `category_other` as a safety finding and
   assigns reviewers on it, so malformed model output must never land there.

## Decision

1. **Provider `llm_judge`, reviewer ID `system_llm_judge`.** A small instruct
   model (Gemma 4 E4B, Q4_0 QAT GGUF; E2B as fallback) served CPU-only by a
   second llama.cpp `llama-server` on the tower, so Llama Guard keeps the GPU.
   `hcms/rating/ktb_judge.py` sends the prompt in `ktb_judge_prompt.txt` plus a
   JSON transcript of the previous four turns and the target message, at
   temperature 0 with a JSON-schema-constrained response, and returns a
   `KtbJudgeVerdict` with `categories` (fixed enum: Suicidal ideation, Suicide
   methods or planning, Suicide rationalization, Self-harm, Substance misuse)
   and a one-sentence `rationale`. The categories alone decide the flag; there
   is no severity score. For assistant replies they describe the reply's own
   behaviour, so a correct crisis reply gets no category.
2. **Stored like the other providers.** One `reviewer_flags` entry per message
   with `categories` and the rationale as `comment`; `category_other` stays
   empty. No schema change anywhere.
3. **Opt-in, own watcher.** `HCMS_MODERATION_PROVIDERS` still defaults to
   `openai_moderation,llama_guard`; the judge runs as `llm_judge` in its own
   watcher per database on the tower, so its latency never sits in front of
   Llama Guard and no existing watcher needs its credentials.
4. **Bounded cycles instead of backfill-then-live.** The watcher repeats: up to
   `MONGODB_LIVE_MAX_EVENTS_PER_CYCLE` live events (or
   `MONGODB_LIVE_MAX_SECONDS_PER_CYCLE`), then at most `MONGODB_RETRY_MAX_PER_CYCLE`
   due retries, then one backfill batch of `MONGODB_BACKFILL_BATCH_SIZE`. Each
   phase is bounded, so sustained traffic cannot starve the backfill or the
   retries, and a live message waits at most one cycle. A failed live rating
   goes into an in-memory retry queue with exponential backoff and is logged as
   an error once `MONGODB_BACKFILL_MAX_RETRIES` attempts are spent; the next
   startup backfill picks it up again.
5. **Resumable, server-filtered backfill.** MongoDB returns only conversations
   that still hold an eligible message without one of the watcher's provider
   entries, ordered by `_id`, and each batch continues from the last complete
   conversation, so a batch costs about what its messages cost. The window
   `MONGODB_BACKFILL_SINCE` (and `--until` of the one-off job) applies to the
   message `timestamp`, never the conversation's creation. History before the
   go-live date is rated once with `hcms.monitoring.backfill_conversations`.
6. **Failures are failures.** Transport errors, truncated completions and
   output that fails schema or semantic validation raise; nothing is written,
   the failure is counted, logged as `ERROR` (the tower's Sentry watcher check
   alerts on it) and retried. A flag is written only while the message still
   has the content that was rated (`expected_content` in the upsert filter); a
   `messages.<i>.content` update re-rates the message with every provider of
   the watcher, so a merged message converges on a rating of its final text.
7. **Minimum bar.** "I want to die", "Ich will sterben", "I want to kill
   myself", "Ich will mich umbringen", "I don't want to live anymore" and "Ich
   möchte nicht mehr leben" must come back with at least one category, alone and
   after benign context: `hcms.evaluation.ktb_judge_eval` (uncached, on the
   tower), the regression test in `tests/test_regression.py` (end-to-end:
   inserts the sentences into the dev database and waits for the deployed judge
   watcher, since the model is reachable only on the tower) and the tower's
   inference probe all enforce it.

## Consequences

- Live detection latency is bounded by the cycle: with a 20 s judge p95, three
  live events, one retry and a batch of two cost at most about two minutes, so
  the judge watcher runs with `MONGODB_BACKFILL_BATCH_SIZE=2`.
- The retry queue is in memory; a crash loses pending retries, and the bounded
  startup backfill recovers them.
- The watchers' summary log now lists counters per enabled provider
  (`<reviewer_id>_success=`, `<reviewer_id>_failure=`, `retry_queue=`).
- Prompt or model changes do not re-rate existing entries; the prompt's SHA-256
  and the model alias are logged by the evaluation so results stay attributable.
  To re-rate, delete the `system_llm_judge` entries and let the backfill run.
- Until the Elderbot backend stops rewriting the whole `messages` array when it
  merges consecutive same-role messages, a merge can still drop a flag written
  in between; with a targeted content write, the content-change path above
  keeps ratings and merges consistent.
- Reviewer assignment on judge flags is enabled in the dashboard only once the
  judge's precision has been measured on a clinician-labelled holdout.
