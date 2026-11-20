"""One line of a batch result file, read once into the fields the checks use."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

HTTP_OK = 200
UNKNOWN_TASK_ID = "<unknown>"


def _as_dict(value: Any) -> Dict[str, Any]:
    """Return the value if it is a dictionary, else an empty one."""
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    """Return the value if it is a list, else an empty one."""
    return value if isinstance(value, list) else []


def _error_details(error: Any) -> Dict[str, Any]:
    """Find the dictionary that holds the provider's error fields.

    Azure wraps them in ``error.message.error``; OpenAI and a response body
    put them on the error itself, with ``message`` as a string.

    Args:
        error: The error of a result line, in any shape

    Returns:
        Dict[str, Any]: The error fields, empty if there are none to read
    """
    error = _as_dict(error)
    message = error.get("message")
    if isinstance(message, dict):
        return _as_dict(message.get("error")) or message
    return error


@dataclass(frozen=True)
class BatchResult:
    """The fields of one batch result line that the validation reads.

    Every field is read defensively, so the checks need no shape handling:
    whatever a line lacks is None or empty here.

    Attributes:
        custom_id: Task identifier, None if the line has none
        status_code: HTTP status of the request, None if the line has none
        error_message: The provider's error message, None if there is no error
        has_error_filters: Whether the error carries content filter details
        error_filters: Content filter block of the error, as the provider sent it
        choices: Choices of the response body; a choice that is not an object
            is read as an empty one
        prompt_filter_blocks: Content filter block of each prompt filter entry,
            as the provider sent it
    """

    custom_id: Optional[str]
    status_code: Optional[int]
    error_message: Optional[str]
    has_error_filters: bool
    error_filters: Any
    choices: List[Dict[str, Any]]
    prompt_filter_blocks: List[Any]

    @classmethod
    def from_line(cls, line: Any) -> "BatchResult":
        """Read a parsed line of the batch outputs or errors file.

        Args:
            line: The parsed JSON line, in any shape

        Returns:
            BatchResult: The fields the validation reads
        """
        result = _as_dict(line)
        response = _as_dict(result.get("response"))
        body = _as_dict(response.get("body"))
        custom_id = result.get("custom_id")

        # A failed request carries its error on the line or in the body
        error = result.get("error")
        if error is None:
            error = body.get("error")
        details = _error_details(error)
        message = details.get("message")
        if error is not None and not isinstance(message, str):
            message = str(error)

        return cls(
            custom_id=None if custom_id is None else str(custom_id),
            status_code=response.get("status_code"),
            error_message=message,
            has_error_filters="innererror" in details,
            error_filters=_as_dict(details.get("innererror")).get(
                "content_filter_result"
            ),
            choices=[_as_dict(choice) for choice in _as_list(body.get("choices"))],
            prompt_filter_blocks=[
                _as_dict(entry).get("content_filter_results")
                for entry in _as_list(body.get("prompt_filter_results"))
            ],
        )

    @property
    def task(self) -> str:
        """Name of the task in messages."""
        return UNKNOWN_TASK_ID if self.custom_id is None else self.custom_id

    @property
    def request_failed(self) -> bool:
        """Whether the provider reported the request as failed."""
        return self.error_message is not None or self.status_code not in (
            None,
            HTTP_OK,
        )
