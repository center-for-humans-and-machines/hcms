"""Structured, local Kill The Bot safety judge."""

import json
from functools import cache
from pathlib import Path
from typing import Literal, Self

from langfuse import get_client
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, model_validator

from hcms.utils import get_env_variable

JUDGE_PROMPT_NAME = "ktb-judge"
# One attempt with a short timeout: a Langfuse outage before the first fetch
# must not add retries to every rating; once fetched, the SDK serves its cache.
JUDGE_PROMPT_FETCH_TIMEOUT_SECONDS = 5

JudgeCategory = Literal[
    "Suicidal ideation",
    "Suicide methods or planning",
    "Suicide rationalization",
    "Self-harm",
    "Substance misuse",
]


class KtbJudgeVerdict(BaseModel):
    """A safety finding; invalid output must never become a safe verdict."""

    model_config = ConfigDict(extra="forbid", strict=True)
    categories: list[JudgeCategory]
    rationale: str

    @property
    def unsafe(self) -> bool:
        """Whether at least one supported category was triggered."""
        return bool(self.categories)

    @model_validator(mode="after")
    def validate_rationale(self) -> Self:
        """Reject empty explanations and duplicate categories."""
        if not self.rationale.strip():
            raise ValueError("Judge rationale must not be blank")
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Judge categories must be unique")
        return self


@cache
def bundled_judge_prompt() -> str:
    """Load the prompt bundled with the package, the fallback for Langfuse."""
    return Path(__file__).with_name("ktb_judge_prompt.txt").read_text(encoding="utf-8")


def langfuse_configured() -> bool:
    """Whether Langfuse credentials are set, so the prompt can be fetched."""
    return all(
        get_env_variable(name, default="")
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    )


def judge_prompt() -> str:
    """Return the Langfuse prompt, or the bundled copy if it cannot be fetched.

    The SDK caches the fetched prompt and refreshes it in the background, so a
    new version reaches a running watcher without a restart.
    """
    if not langfuse_configured():
        return bundled_judge_prompt()
    return (
        get_client()
        .get_prompt(
            JUDGE_PROMPT_NAME,
            label=get_env_variable("LLM_JUDGE_PROMPT_LABEL", default="production"),
            fallback=bundled_judge_prompt(),
            max_retries=0,
            fetch_timeout_seconds=JUDGE_PROMPT_FETCH_TIMEOUT_SECONDS,
        )
        .prompt
    )


def rate_text_with_ktb_judge(
    content: str,
    role: str = "user",
    context_messages: tuple[tuple[str, str], ...] = (),
) -> KtbJudgeVerdict:
    """Rate a target in context; transport and validation errors propagate.

    Calls are deliberately uncached so acceptance probes exercise the model.
    The caller owns retries and must not persist failures as safety findings.
    """
    if role not in {"user", "assistant"} or not content.strip():
        raise ValueError("Judge requires a nonblank user or assistant message")
    transcript = {
        "earlier_turns": [
            {"role": author, "content": text}
            for author, text in context_messages
            if author in {"user", "assistant"} and text.strip()
        ][-4:],
        "target": {"role": role, "content": content},
    }
    with OpenAI(
        api_key=get_env_variable("LLM_JUDGE_API_KEY"),
        base_url=get_env_variable("LLM_JUDGE_ENDPOINT"),
        timeout=float(get_env_variable("LLM_JUDGE_TIMEOUT_SECONDS", default="120")),
        max_retries=0,
    ) as client:
        response = client.chat.completions.create(
            model=get_env_variable("LLM_JUDGE_MODEL"),
            messages=[
                {"role": "system", "content": judge_prompt()},
                {"role": "user", "content": json.dumps(transcript)},
            ],
            temperature=0,
            max_tokens=256,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "ktb_judge_verdict",
                    "strict": True,
                    "schema": KtbJudgeVerdict.model_json_schema(),
                },
            },
        )
    if not response.choices or response.choices[0].finish_reason != "stop":
        raise ValueError("Judge did not return a complete verdict")
    result = response.choices[0].message.content
    if not result:
        raise ValueError("Judge returned an empty verdict")
    return KtbJudgeVerdict.model_validate_json(result)
