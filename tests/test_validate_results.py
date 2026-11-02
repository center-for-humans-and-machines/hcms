"""Unit tests for BaseProcessor.validate_results on synthetic batch results.

No network: the API client is stubbed and the result dicts mirror lines from
the output and error files of a real Azure batch run (BAN-1382).
"""

# pylint: disable=missing-function-docstring,protected-access

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from hcms.monitoring.api import BatchConfig
from hcms.monitoring.processors import batch as batch_module
from hcms.monitoring.processors.base import BaseProcessor

AZURE_ENDPOINT = "https://example.openai.azure.com/"
OPENAI_ENDPOINT = "https://api.openai.com/"

SAFE_PROMPT_FILTERS: Dict[str, Any] = {
    "hate": {"filtered": False, "severity": "safe"},
    "jailbreak": {"detected": False, "filtered": False},
    "self_harm": {"filtered": False, "severity": "safe"},
    "sexual": {"filtered": False, "severity": "safe"},
    "violence": {"filtered": False, "severity": "safe"},
}
SAFE_CHOICE_FILTERS: Dict[str, Any] = {
    "protected_material_code": {"detected": False, "filtered": False}
}
SELF_HARM_FILTERS: Dict[str, Any] = {
    **SAFE_PROMPT_FILTERS,
    "self_harm": {"filtered": True, "severity": "medium"},
}
FILTERED_MESSAGE = (
    "The response was filtered due to the prompt triggering Azure OpenAI's "
    "content management policy. Please modify your prompt and retry."
)


def _output(
    custom_id: str,
    choice_filters: Optional[Dict[str, Any]] = None,
    prompt_filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A line from the batch outputs file, as Azure writes it."""
    return {
        "custom_id": custom_id,
        "response": {
            "body": {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                        "content_filter_results": choice_filters,
                    }
                ],
                "prompt_filter_results": [
                    {"prompt_index": 0, "content_filter_results": prompt_filters}
                ],
            },
            "request_id": "7a2b52b5",
            "status_code": 200,
        },
        "error": None,
    }


def _good_output(custom_id: str) -> Dict[str, Any]:
    return _output(custom_id, SAFE_CHOICE_FILTERS, SAFE_PROMPT_FILTERS)


def _filtered_error(
    custom_id: str, content_filter_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """A line from the batch errors file for a prompt Azure rejected."""
    return {
        "custom_id": custom_id,
        "response": {"request_id": "", "status_code": 400},
        "error": {
            "code": None,
            "message": {
                "error": {
                    "message": FILTERED_MESSAGE,
                    "type": None,
                    "param": "prompt",
                    "code": "content_filter",
                    "status": 400,
                    "innererror": {
                        "code": "ResponsibleAIPolicyViolation",
                        "content_filter_result": content_filter_result,
                    },
                }
            },
        },
    }


def _processor(
    tmp_path: Path, monkeypatch, endpoint: str, num_rows: int
) -> BaseProcessor:
    monkeypatch.setattr(batch_module, "create_client", object)
    input_file = tmp_path / "prompts.json"
    input_file.write_text("[]", encoding="utf-8")
    config = BatchConfig(
        system_prompt="You are a companion.",
        input_file=input_file,
        output_dir=tmp_path / "batch",
        endpoint=endpoint,
        batch_model="gpt-4o-global-batch",
        chat_completions_model="gpt-4o-2024-08-06",
    )
    processor = BaseProcessor(
        config, input_column_name="question", output_column_name="answer"
    )
    processor._num_rows = num_rows
    return processor


def test_azure_results_with_filter_blocks_pass(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=2)

    assert processor.validate_results([_good_output("task-0"), _good_output("task-1")])


def test_null_choice_filter_block_is_a_validation_error(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)
    result = _output("task-5", None, SAFE_PROMPT_FILTERS)

    with pytest.raises(ValueError) as excinfo:
        processor.validate_results([result])

    assert "Missing completion content filter results in task task-5" in str(
        excinfo.value
    )


def _null_prompt_block(result: Dict[str, Any]) -> None:
    prompt_filter_results = result["response"]["body"]["prompt_filter_results"]
    prompt_filter_results[0]["content_filter_results"] = None


def _null_prompt_list(result: Dict[str, Any]) -> None:
    result["response"]["body"]["prompt_filter_results"] = None


def _empty_prompt_list(result: Dict[str, Any]) -> None:
    result["response"]["body"]["prompt_filter_results"] = []


@pytest.mark.parametrize(
    "mutate", [_null_prompt_block, _null_prompt_list, _empty_prompt_list]
)
def test_missing_prompt_filter_block_is_a_validation_error(
    tmp_path, monkeypatch, mutate: Callable[[Dict[str, Any]], None]
):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)
    result = _good_output("task-7")
    mutate(result)

    with pytest.raises(ValueError) as excinfo:
        processor.validate_results([result])

    assert "Missing prompt content filter results in task task-7" in str(excinfo.value)


def test_rejected_prompt_is_reported_as_a_finding(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)

    with pytest.raises(ValueError) as excinfo:
        processor.validate_results([_filtered_error("task-35", SELF_HARM_FILTERS)])

    message = str(excinfo.value)
    assert "Failed request for task task-35: status code 400" in message
    assert f"Error in task task-35: {FILTERED_MESSAGE}" in message


def test_error_entry_filters_are_checked(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)

    with pytest.raises(ValueError, match="self_harm filter triggered in task task-35"):
        processor._validate_azure_specific(
            _filtered_error("task-35", SELF_HARM_FILTERS)
        )


def test_null_error_entry_filters_are_a_validation_error(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)

    with pytest.raises(
        ValueError, match="Missing error content filter results in task task-35"
    ):
        processor._validate_azure_specific(_filtered_error("task-35", None))


def test_live_run_shape_reports_findings_instead_of_crashing(tmp_path, monkeypatch):
    """100 inputs, 99 results: 89 clean, 9 with a null choice block, 1 rejected."""
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=100)
    null_ids = [f"task-{i}" for i in (5, 62, 34, 21, 77, 25, 74, 95, 96)]
    results: List[Dict[str, Any]] = [
        _good_output(f"task-{i}") for i in range(100) if f"task-{i}" not in null_ids
    ][:89]
    results += [_output(custom_id, None, SAFE_PROMPT_FILTERS) for custom_id in null_ids]
    results.append(_filtered_error("task-35", SELF_HARM_FILTERS))
    assert len(results) == 99

    with pytest.raises(ValueError) as excinfo:
        processor.validate_results(copy.deepcopy(results))

    message = str(excinfo.value)
    assert "Number of results (99) does not match number of inputs to batch (100)" in (
        message
    )
    assert "Number of successful tasks: 89" in message
    assert "Number of failed tasks: 10" in message
    for custom_id in null_ids:
        assert f"Missing completion content filter results in task {custom_id}" in (
            message
        )
    assert "Error in task task-35" in message


def test_openai_endpoint_skips_azure_filter_checks(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, OPENAI_ENDPOINT, num_rows=1)

    assert processor.validate_results([_output("task-0", None, None)])
