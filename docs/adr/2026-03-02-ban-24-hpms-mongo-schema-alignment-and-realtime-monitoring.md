<!-- markdownlint-disable MD022 MD032 -->

# HPMS MongoDB Schema and Realtime Monitoring Spec

## Objective
Define the HPMS backend changes required to:
1. Use the same MongoDB conversation schema as the dashboard.
2. Watch MongoDB for new message additions.
3. Rate each new message with:
   - `_rate_text_with_openai_moderation`
   - `_rate_text_with_llama_guard`
4. Persist monitoring outputs using only the fields in the provided schema.

## Scope
- In scope:
  - HPMS database/document models for conversation data.
  - MongoDB watcher service (change stream + startup backfill).
  - Persistence logic for moderation outputs in message-level fields.
  - Decision on `MessageDocument` vs `ConversationMessage`.
- Out of scope:
  - Dashboard UI.
  - Reviewer assignment cron logic implementation.
  - Legacy/backward compatibility support.

## Canonical Schema (Exact Contract)
HPMS and dashboard must share this schema:

```json
{
  "_id": "ObjectId",
  "participant_id": "string",
  "model": "string",
  "experiment_id": "string",
  "created_at": "ISODate",
  "messages": [
    {
      "content": "string",
      "role": "string",
      "timestamp": "ISODate",
      "type": "string",
      "user_flag": {
        "category": "string",
        "category_other": "string",
        "reviews": [
          {
            "reviewer_id": "string",
            "approved": "bool",
            "comment": "string",
            "reviewed_at": "ISODate"
          }
        ]
      },
      "reviewer_flags": [
        {
          "reviewer_id": "string",
          "created_at": "ISODate",
          "categories": ["string"],
          "category_other": "string"
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
  "reviewed_by": [
    {
      "reviewer_id": "string",
      "reviewed_at": "ISODate"
    }
  ],
  "assigned_to": [
    {
      "reviewer_id": "string",
      "reason": "string",
      "assigned_at": "ISODate"
    }
  ]
}
```

No additional schema fields are introduced.

## Monitoring Output Storage (Using Only Provided Fields)
Because no dedicated monitoring field exists in the schema, monitoring results are stored in `messages[].reviewer_flags`.

### System reviewer IDs
- OpenAI moderation result entry:
  - `reviewer_id = "system_openai_moderation"`
- Llama Guard result entry:
  - `reviewer_id = "system_llama_guard"`

### Stored values
Each system result is one `reviewer_flags` element:
- `reviewer_id`: system reviewer ID above.
- `created_at`: rating timestamp.
- `categories`:
  - OpenAI: list of flagged category names from `_rate_text_with_openai_moderation` output.
  - Llama Guard: empty list for safe (`"0"`) or single-item list with mapped category for unsafe.
- `category_other`:
  - Empty string by default.
  - If provider returns unexpected free-form output, store it in `category_other`.

### Idempotency rule
Enforce one system entry per provider per message by upserting on `(conversation_id, message_index, reviewer_id)`.

## Decision: `MessageDocument` vs `ConversationMessage`
Keep both.

- Keep `ConversationMessage` in `hpms/loading/models.py` for offline/synthetic dataset processing.
- Add `MessageDocument` (Mongo conversation message model) for dashboard-aligned database documents.

Reason:
- They represent different domains and field sets.
- `ConversationMessage` currently supports dataset generation/evaluation shape.
- Mongo conversation messages need flag/review fields from the dashboard contract.

## Architecture Changes

### 1) New database models
Add a Mongo schema module, e.g.:
- `hpms/database/models.py`

Models to add:
- `UserFlagReviewDocument`
- `UserFlagDocument`
- `ReviewerFlagDocument`
- `MessageDocument`
- `ConversationReviewStateDocument` (for `opened_by` / `reviewed_by` elements)
- `ConversationAssignmentDocument` (for `assigned_to` elements)
- `ConversationDocument`

These models must match the canonical schema above exactly.

### 2) New Mongo repository layer
Add:
- `hpms/database/repository.py`

Responsibilities:
- Connect to MongoDB collection.
- Fetch conversations/messages needing monitoring writes.
- Upsert system reviewer flags for each provider.
- Keep writes idempotent.

### 3) Realtime watcher service
Add:
- `hpms/monitoring/realtime_watcher.py`

Responsibilities:
- On startup: backfill messages missing either system reviewer entry.
- Start MongoDB change stream on `Conversations` collection.
- Process new message additions and message content updates.
- Call both rating functions.
- Persist results into `messages[].reviewer_flags`.

### 4) Runtime entrypoint
Add script:
- `hpms/monitoring/watch_mongo_conversations.py`

Responsibilities:
- Read env config.
- Start watcher loop.
- Handle graceful shutdown.

### 5) Configuration
Add env vars to `env.example`:
- `MONGODB_URI`
- `MONGODB_DATABASE`
- `MONGODB_COLLECTION=Conversations`
- `MONGODB_CHANGE_STREAM_MAX_AWAIT_MS=1000`
- `MONGODB_BACKFILL_BATCH_SIZE=200`

## Processing Rules

### Message selection
A message requires processing if either of the following does not exist in `reviewer_flags`:
- `reviewer_id = "system_openai_moderation"`
- `reviewer_id = "system_llama_guard"`

Validation tolerance:
- if `messages[].user_flag` is missing, watcher defaults it to an empty object (`category=""`, `category_other=""`, `reviews=[]`) before processing.

### OpenAI normalization
Input: list from `_rate_text_with_openai_moderation`.
- Collect non-zero/string category entries into `categories`.
- On unexpected output, fallback:
  - `categories=[]`
  - `category_other` set to serialized raw output.

### Llama Guard normalization
Input: string from `_rate_text_with_llama_guard`.
- If output is `"0"`: `categories=[]`, `category_other=""`.
- If output matches known unsafe category: `categories=[<category>]`, `category_other=""`.
- Otherwise: `categories=[]`, `category_other=<raw output>`.

### Failure behavior
- If one provider fails, still persist the other provider result.
- Retry failed provider on next change event or next startup backfill.
- Do not create duplicate system reviewer entries.

## Testing Plan

### Unit tests
1. OpenAI result normalization.
2. Llama Guard result normalization.
3. Idempotent upsert behavior for system reviewer IDs.
4. Message needs-processing detection logic.

### Integration tests
1. Startup backfill rates existing unrated messages.
2. Change stream insert/update event processes only affected messages.
3. Partial failure retry path.
4. No duplicate `reviewer_flags` entries after repeated identical events.

## Acceptance Criteria
1. HPMS writes and validates conversation documents using the exact schema above.
2. When a new message is added, both rating functions are executed.
3. Each provider output is saved in the corresponding message under `reviewer_flags` using system reviewer IDs.
4. No non-schema fields are written.
5. `ConversationMessage` remains for dataset workflows; new Mongo-specific `MessageDocument` is used for DB workflows.

## File-Level Change Plan
- `pyproject.toml`: add Mongo client dependency.
- `env.example`: add Mongo configuration vars.
- `hpms/database/models.py`: add canonical Mongo document models.
- `hpms/database/repository.py`: add DB read/write methods.
- `hpms/monitoring/realtime_watcher.py`: add backfill + change stream processing.
- `hpms/monitoring/watch_mongo_conversations.py`: add module entrypoint runner.
- `tests/`: add watcher/repository normalization and idempotency tests.
