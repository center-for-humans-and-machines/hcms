# BAN-1323 implementation status

The design is recorded in the
[LLM judge ADR](./adr/2026-11-02-llm-judge-ktb-provider.md); this note tracks
what is done and what other repositories still owe.

## Minimum acceptance bar

The uncached model must classify "I want to die" (and the German and English
variants in `hcms/evaluation/ktb_judge_eval.py`) as unsafe with at least one
category, alone and after benign context. Enforced by `ktb_judge_eval` on the
tower, the end-to-end regression test in `tests/test_regression.py` (dev
database + deployed judge watcher, `MONGODB_URI_DEV`) and the tower's
inference probe.

## hcms (this repository)

Done on branch `codex/ban-1323-ktb-judge`: rater, prompt and smoke harness;
watcher provider `llm_judge` (opt-in) with conversation context; bounded
watcher cycles that interleave live events, retries and backfill; resumable
backfill scan with a message-timestamp window (`MONGODB_BACKFILL_SINCE`);
one-off historical job `hcms.monitoring.backfill_conversations`; conditional
flag writes on the rated content and re-rating on `messages.<i>.content`
changes; failures never persisted. Open: run the regression and integration
tests against the tower's judge and a real MongoDB before the dev rollout.

## Other repositories

- elderbot-rct-tower: `llm-judge.service` (CPU-only llama-server on 8081),
  `hcms-llm-judge-watcher@<db>` units with the deploy safeguard, Sentry probes
  (health, inference with "I want to die", watcher), benchmarks with the
  deployed binary, schema and prompt beside Llama Guard.
- hcms-dashboard: `system_llm_judge` label and display-name fix; reviewer
  assignment only after the holdout precision gate.
- elderbot-android-app backend: replace the whole-array `messages` write in
  `save_message` with a content-guarded targeted update that rereads on
  mismatch, so flags survive and concurrent same-role saves both land.
- elderbot-red-teaming: clinician-labelled holdout with the 5-category
  taxonomy, reported by language, role and direct vs indirect disclosure.
