"""Unit tests for BaseProcessor.validate_results on synthetic batch results.

No network: the API client is stubbed and the result dicts mirror lines from
the output and error files of a real Azure batch run (BAN-1382).
"""

# pylint: disable=missing-function-docstring,protected-access

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from hcms.monitoring.api import BatchConfig
from hcms.monitoring.processors import batch as batch_module
from hcms.monitoring.processors.base import BaseProcessor

AZURE_ENDPOINT = "https://example.openai.azure.com/"
OPENAI_ENDPOINT = "https://api.openai.com/"
UNKNOWN_ENDPOINT = "https://api.together.xyz/"

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
SERVER_ERROR_MESSAGE = "The server had an error while processing your request."


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


def _error_entry(
    custom_id: str, status_code: int, error: Dict[str, Any]
) -> Dict[str, Any]:
    """A line from the batch errors file."""
    return {
        "custom_id": custom_id,
        "response": {"request_id": "", "status_code": status_code},
        "error": {"code": None, "message": {"error": error}},
    }


def _filtered_error(
    custom_id: str, content_filter_result: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """The error line for a prompt Azure rejected."""
    return _error_entry(
        custom_id,
        400,
        {
            "message": FILTERED_MESSAGE,
            "type": None,
            "param": "prompt",
            "code": "content_filter",
            "status": 400,
            "innererror": {
                "code": "ResponsibleAIPolicyViolation",
                "content_filter_result": content_filter_result,
            },
        },
    )


def _server_error(custom_id: str) -> Dict[str, Any]:
    """The error line for a request that failed without filter details."""
    return _error_entry(
        custom_id,
        500,
        {"message": SERVER_ERROR_MESSAGE, "type": "server_error", "code": None},
    )


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


def _validation_message(processor: BaseProcessor, results: List[Dict[str, Any]]) -> str:
    with pytest.raises(ValueError) as excinfo:
        processor.validate_results(results)
    return str(excinfo.value)


def test_azure_results_with_filter_blocks_pass(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=2)

    assert processor.validate_results([_good_output("task-0"), _good_output("task-1")])


def test_empty_filter_block_lists_nothing_to_check(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)

    assert processor.validate_results([_output("task-0", {}, SAFE_PROMPT_FILTERS)])


def test_null_choice_filter_block_is_a_validation_error(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)

    message = _validation_message(
        processor, [_output("task-5", None, SAFE_PROMPT_FILTERS)]
    )

    assert "Missing completion content filter results in task task-5" in message


def _null_prompt_block(result: Dict[str, Any]) -> None:
    prompt_filter_results = result["response"]["body"]["prompt_filter_results"]
    prompt_filter_results[0]["content_filter_results"] = None


def _null_prompt_list(result: Dict[str, Any]) -> None:
    result["response"]["body"]["prompt_filter_results"] = None


def _empty_prompt_list(result: Dict[str, Any]) -> None:
    result["response"]["body"]["prompt_filter_results"] = []


def _null_prompt_entry(result: Dict[str, Any]) -> None:
    result["response"]["body"]["prompt_filter_results"] = [None]


@pytest.mark.parametrize(
    "mutate",
    [_null_prompt_block, _null_prompt_list, _empty_prompt_list, _null_prompt_entry],
)
def test_missing_prompt_filter_block_is_a_validation_error(
    tmp_path, monkeypatch, mutate: Callable[[Dict[str, Any]], None]
):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)
    result = _good_output("task-7")
    mutate(result)

    message = _validation_message(processor, [result])

    assert "Missing prompt content filter results in task task-7" in message


def test_null_filter_entry_names_the_filter(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)
    choice_filters = {**SAFE_CHOICE_FILTERS, "hate": None}

    message = _validation_message(
        processor, [_output("task-3", choice_filters, SAFE_PROMPT_FILTERS)]
    )

    assert "Missing hate filter result in task task-3" in message


def test_rejected_prompt_reports_status_message_and_filter(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)

    message = _validation_message(
        processor, [_filtered_error("task-35", SELF_HARM_FILTERS)]
    )

    assert "Failed request for task task-35: status code 400" in message
    assert f"Error in task task-35: {FILTERED_MESSAGE}" in message
    assert "self_harm filter triggered in task task-35" in message
    assert "Number of failed tasks: 1" in message


def test_rejected_prompt_without_filter_details_is_a_validation_error(
    tmp_path, monkeypatch
):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)

    message = _validation_message(processor, [_filtered_error("task-35", None)])

    assert "Missing error content filter results in task task-35" in message


def test_server_error_entry_reports_only_the_request_failure(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=1)

    message = _validation_message(processor, [_server_error("task-8")])

    assert "Failed request for task task-8: status code 500" in message
    assert f"Error in task task-8: {SERVER_ERROR_MESSAGE}" in message
    assert "Missing" not in message


def test_count_mismatch_is_reported_with_the_summary(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=2)

    message = _validation_message(processor, [_good_output("task-0")])

    assert "Number of results (1) does not match number of inputs to batch (2)" in (
        message
    )
    assert "Number of successful tasks: 1" in message
    assert "Number of failed tasks: 0" in message


def test_live_run_shape_reports_every_finding(tmp_path, monkeypatch):
    """100 inputs, 100 results: 90 clean, 9 with a null choice block, 1 rejected."""
    processor = _processor(tmp_path, monkeypatch, AZURE_ENDPOINT, num_rows=100)
    null_ids = [f"task-{i}" for i in (5, 62, 34, 21, 77, 25, 74, 95, 96)]
    rejected_id = "task-35"
    failed_ids = null_ids + [rejected_id]
    results: List[Dict[str, Any]] = [
        _good_output(f"task-{i}") for i in range(100) if f"task-{i}" not in failed_ids
    ]
    results += [_output(custom_id, None, SAFE_PROMPT_FILTERS) for custom_id in null_ids]
    results.append(_filtered_error(rejected_id, SELF_HARM_FILTERS))
    assert len(results) == 100

    message = _validation_message(processor, results)

    assert "does not match" not in message
    assert "Number of successful tasks: 90" in message
    assert "Number of failed tasks: 10" in message
    for custom_id in null_ids:
        assert f"Missing completion content filter results in task {custom_id}" in (
            message
        )
    assert f"Error in task {rejected_id}" in message
    assert f"self_harm filter triggered in task {rejected_id}" in message


def test_openai_endpoint_skips_azure_filter_checks(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, OPENAI_ENDPOINT, num_rows=1)

    assert processor.validate_results([_output("task-0", None, None)])


def test_unknown_endpoint_is_rejected_before_any_result(tmp_path, monkeypatch):
    processor = _processor(tmp_path, monkeypatch, UNKNOWN_ENDPOINT, num_rows=0)

    with pytest.raises(ValueError, match="Invalid endpoint type"):
        processor.validate_results([])
