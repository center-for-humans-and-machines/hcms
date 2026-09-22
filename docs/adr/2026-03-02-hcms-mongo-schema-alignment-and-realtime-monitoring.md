<!-- markdownlint-disable MD022 MD032 -->

# HCMS MongoDB Schema and Realtime Monitoring Spec

## Objective

Define the HCMS backend changes required to:

1. Use the same MongoDB conversation schema as the dashboard.
2. Watch MongoDB for new message additions.
3. Rate each new message with:
   - `_rate_text_with_openai_moderation`
   - `_rate_text_with_llama_guard`
4. Persist monitoring outputs using only the fields in the provided schema.

## Scope

- In scope:
  - HCMS database/document models for conversation data.
  - MongoDB watcher service (change stream + startup backfill).
  - Persistence logic for moderation outputs in message-level fields.
  - Decision on `MessageDocument` vs `ConversationMessage`.
- Out of scope:
  - Dashboard UI.
  - Reviewer assignment cron logic implementation.
  - Legacy/backward compatibility support.

## Canonical Schema (Exact Contract)

HCMS and dashboard must share this schema:

```json
{
  "_id": "ObjectId",
  "conversation_id": "string",
  "participant_id": "string",
  "model": "string",
  "experiment_id": "string",
  "project_id": "string",
  "created_at": "ISODate",
  "custom_system_message_id": "string",
  "multi_rounds": "boolean",
  "messages": [
    {
      "content": "string",
      "role": "string",
      "timestamp": "ISODate",
      "type": "string",
      "flagged": "boolean",
      "flagged_at": "ISODate",
      "flagged_by": "string",
      "flag_category": "string",
      "flag_other_reason": "string",
      "user_flag": {
        "category": "string",
        "category_other": "string",
        "reviews": [
          {
            "reviewer_id": "string",
            "reviewer_username": "string",
            "approved": "boolean",
            "comment": "string",
            "reviewed_at": "ISODate"
          }
        ]
      },
      "reviewer_flags": [
        {
          "reviewer_id": "string",
          "reviewer_by_username": "string",
          "created_at": "ISODate",
          "categories": ["string"],
          "category_other": "string",
          "comment": "string"
        }
      ],
      "duplicate_flags": [
        {
          "reviewer_id": "string",
          "reviewer_username": "string",
          "flagged_at": "ISODate"
        }
      ]
    }
  ],
  "opened_by": [
    {
      "reviewer_id": "string",
      "opened_at": "ISODate"
    }
  ],
  "assigned_messages": [
    {
      "reviewer_id": "string",
      "message_index": "number",
      "reason": "random_sample | participant_flag | expert_escalation",
      "assigned_at": "ISODate"
    }
  ],
  "reviewed_messages": [
    {
      "reviewer_id": "string",
      "message_index": "number",
      "reviewed_at": "ISODate"
    }
  ]
}
```

No additional schema fields are introduced.

## Monitoring Output Storage (Using Only Provided Fields)

Because no dedicated monitoring field exists in the schema, monitoring results are
stored in `messages[].reviewer_flags`.

### System reviewer IDs

- OpenAI moderation result entry:
  - `reviewer_id = "system_openai_moderation"`
  - `reviewer_by_username = "system_openai_moderation"`
- Llama Guard result entry:
  - `reviewer_id = "system_llama_guard"`
  - `reviewer_by_username = "system_llama_guard"`

### Stored values

Each system result is one `reviewer_flags` element:

- `reviewer_id`: system reviewer ID above.
- `reviewer_by_username`: same value as the system reviewer ID.
- `created_at`: rating timestamp.
- `categories`:
  - OpenAI: list of flagged category names from
    `_rate_text_with_openai_moderation` output.
  - Llama Guard: empty list for safe (`"0"`) or single-item list with mapped
    category for unsafe.
- `category_other`:
  - Empty string by default.
  - If provider returns unexpected free-form output, store it in
    `category_other`.
- `comment`:
  - Empty string for automated writes.

### Idempotency rule

Enforce one system entry per provider per message by upserting on
`(_id, message_index, reviewer_id)`.

## Decision: `MessageDocument` vs `ConversationMessage`

Keep both.

- Keep `ConversationMessage` in `hcms/loading/models.py` for offline/synthetic
  dataset processing.
- Add `MessageDocument` (Mongo conversation message model) for
  dashboard-aligned database documents.

Reason:

- They represent different domains and field sets.
- `ConversationMessage` currently supports dataset generation/evaluation shape.
- Mongo conversation messages need flag/review fields from the dashboard
  contract.

## Architecture Changes

### 1) New database models

Add a Mongo schema module:

- `hcms/database/models.py`

Models to add:

- `UserFlagReviewDocument`
- `UserFlagDocument`
- `ReviewerFlagDocument`
- `DuplicateFlagDocument`
- `MessageDocument`
- `OpenedByDocument`
- `AssignedMessageDocument`
- `ReviewedMessageDocument`
- `ConversationDocument`

These models must match the canonical schema above exactly.

### 2) Mongo repository layer

`hcms/database/repository.py` is responsible for:

- Connecting to MongoDB collection.
- Fetching conversations/messages needing monitoring writes.
- Upserting system reviewer flags for each provider.
- Keeping writes idempotent using Mongo `_id`.

### 3) Realtime watcher service

`hcms/monitoring/realtime_watcher.py` is responsible for:

- Backfilling messages missing either system reviewer entry on startup.
- Watching the `Conversations` collection via MongoDB change streams.
- Reconnecting with exponential backoff + jitter after change stream failures.
- Processing new message additions and message content updates.
- Calling both rating functions.
- Persisting results into `messages[].reviewer_flags`.
- Logging runtime counters for observability.

### 4) Runtime entrypoint

`hcms/monitoring/watch_mongo_conversations.py` is responsible for:

- Reading env config.
- Configuring watcher log level from env.
- Starting the watcher loop.
- Handling graceful shutdown.

### 5) Configuration

Add env vars to `env.example`:

- `MONGODB_URI`
- `MONGODB_DATABASE`
- `MONGODB_COLLECTION=Conversations`
- `MONGODB_CHANGE_STREAM_MAX_AWAIT_MS=1000`
- `MONGODB_BACKFILL_BATCH_SIZE=200`
- `MONGODB_BACKFILL_MAX_RETRIES=10`
- `MONGODB_BACKFILL_RETRY_SLEEP_SECONDS=2`
- `MONGODB_RECONNECT_BACKOFF_BASE_SECONDS=1`
- `MONGODB_RECONNECT_BACKOFF_MAX_SECONDS=30`
- `MONGODB_RECONNECT_BACKOFF_JITTER_SECONDS=0.25`
- `HCMS_LOG_LEVEL=INFO`

## Processing Rules

### Message selection

A message requires processing if either of the following does not exist in
`reviewer_flags`:

- `reviewer_id = "system_openai_moderation"`
- `reviewer_id = "system_llama_guard"`

The schema is strict for fresh databases:

- producers must populate `user_flag`, `reviewer_flags`, `duplicate_flags`,
  `assigned_messages`, and `reviewed_messages`;
- HCMS does not accept legacy field names or compatibility aliases.

### OpenAI normalization

Input: list from `_rate_text_with_openai_moderation`.

- Collect non-zero/string category entries into `categories`.
- On unexpected output, fallback:
  - `categories=[]`
  - `category_other` set to serialized raw output.

### Llama Guard normalization

Input: string from `_rate_text_with_llama_guard`.

- If output is `"0"`: `categories=[]`, `category_other=""`.
- If output matches known unsafe category: `categories=[<category>]`,
  `category_other=""`.
- Otherwise: `categories=[]`, `category_other=<raw output>`.

### Failure behavior

- If one provider fails, still persist the other provider result.
- During startup backfill, retry failed provider writes until success, retry
  budget exhaustion, or no retry-eligible targets remain.
- After startup, retry failed provider on future change events and future
  restarts.
- Do not create duplicate system reviewer entries.

### Import behavior

- `hcms.monitoring` exports `RealtimeConversationWatcher` lazily.
- Importing non-watcher monitoring components must not require moderation
  credentials at module import time.

## Testing Plan

### Unit tests

1. OpenAI result normalization.
2. Llama Guard result normalization.
3. Idempotent upsert behavior for system reviewer IDs.
4. Message needs-processing detection logic.
5. Strict rejection of legacy field names and invalid message-level assignment
   state.

### Integration tests

1. Startup backfill rates existing unrated messages.
2. Change stream insert/update event processes only affected messages.
3. Partial failure retry path.
4. No duplicate `reviewer_flags` entries after repeated identical events.

## Acceptance Criteria

1. HCMS writes and validates conversation documents using the exact schema
   above.
2. When a new message is added, both rating functions are executed.
3. Each provider output is saved in the corresponding message under
   `reviewer_flags` using system reviewer IDs.
4. No non-schema fields are written.
5. `ConversationMessage` remains for dataset workflows; new Mongo-specific
   `MessageDocument` is used for DB workflows.
