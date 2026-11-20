"""Base class for processing data using batch processing."""

from abc import ABC
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import polars as pl
from pydantic import BaseModel

from hcms.constants import CONTENT, ROLE, SYSTEM, USER
from hcms.monitoring.api import BatchConfig
from hcms.monitoring.processors.batch import BatchProcessor
from hcms.monitoring.processors.batch_result import HTTP_OK, BatchResult

ACCEPTED_SEVERITIES = ("safe", "low")


class BaseProcessor(ABC):
    """Base class for processing data using batch processing."""

    def __init__(
        self, config: BatchConfig, input_column_name: str, output_column_name: str
    ):
        # Initialize configuration
        self.input_column_name: str = input_column_name
        self.output_column_name: str = output_column_name
        self.config: BatchConfig = config

        # Initialize with default value
        self._num_rows: int = 0

        # Initialize batch processor
        self._batch_processor: BatchProcessor = BatchProcessor(config)

    def process_samples(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Process first n samples through direct API.

        Args:
            num_samples: Number of samples to process

        Returns:
            List[Dict]: Processing results
        """
        df: pl.DataFrame = self.config.data_loader(self.config.input_file)
        return self._process_data(df.head(num_samples))

    def _process_data(self, df: pl.DataFrame) -> List[Dict[str, Any]]:
        """Process regression test data through direct API calls.

        Args:
            df: DataFrame with regression test data

        Returns:
            List[Dict]: Processing results for samples

        """
        results = []
        for row in df.iter_rows(named=True):
            # Request completion
            response = self._batch_processor.client.chat.completions.create(
                model=self.config.chat_completions_model,
                temperature=self.config.temperature,
                messages=[
                    {ROLE: SYSTEM, CONTENT: self.config.system_prompt},
                    {ROLE: USER, CONTENT: row[self.input_column_name]},
                ],
            )

            # Safely get content filter results
            try:
                message_content_filter_results = response.choices[
                    0
                ].content_filter_results
            except (AttributeError, IndexError):
                message_content_filter_results = {}

            try:
                prompt_content_filter_results = response.prompt_filter_results[0][
                    "content_filter_results"
                ]
            except (AttributeError, IndexError, KeyError):
                prompt_content_filter_results = {}

            # Append results
            results.append(
                {
                    self.input_column_name: row[self.input_column_name],
                    self.output_column_name: response.choices[0].message.content,
                    "content_filters": {
                        "message": message_content_filter_results,
                        "prompt": prompt_content_filter_results,
                    },
                }
            )
        return results

    def process_batch(self) -> List[Dict[str, Any]]:
        """Process full dataset through batch API.

        The results of the batch processing will be retrieved asynchronously.
        The results are saved in the output file as defined in config.output_dir.

        Returns:
            List[Dict]: Processing results
        """
        # Load data
        df: pl.DataFrame = self.config.data_loader(self.config.input_file)

        # Store number of rows for validation
        self._num_rows = len(df)

        response_format_model: Optional[Type[BaseModel]] = None
        if self.config.response_format_model:
            # Check if response format model is valid
            if not issubclass(self.config.response_format_model, BaseModel):
                raise ValueError(
                    "response_format_model must be an instance of BaseModel."
                )
            response_format_model = self.config.response_format_model

        batch_file: Path = self._batch_processor.create_batch_file(
            df=df,
            column_name=self.input_column_name,
            response_format_model=response_format_model,
        )

        # Submit batch job (inputs)
        job_id: str = self._batch_processor.submit_batch(batch_file)

        # Get results (outputs and errors)
        results: List[Dict[str, Any]] = self._batch_processor.get_results(job_id)

        return results

    def validate_results(self, results: List[Dict[str, Any]]) -> bool:
        """Validate results of the batch processing.

        This method should be called after the batch processing is completed.

        Args:
            results: List of processing results

        Returns:
            bool: True if all results are valid

        Raises:
            ValueError: If any result fails validation or row count mismatch,
                or if the endpoint is neither Azure nor OpenAI
        """
        if not (self.config.is_endpoint_azure or self.config.is_endpoint_openai):
            raise ValueError("Invalid endpoint type. Must be either Azure or OpenAI.")

        validation_errors: List[str] = []
        failed_task_ids: set[str] = set()

        # Validate row count
        if len(results) != self._num_rows:
            validation_errors.append(
                f"Number of results ({len(results)}) does not match "
                f"number of inputs to batch ({self._num_rows})\n"
            )

        # Read the shape of every line once; the checks only see BatchResult
        batch_results = [BatchResult.from_line(line) for line in results]
        for position, result in enumerate(batch_results, start=1):
            task_errors = self._validate_result(result)
            if result.custom_id is None:
                validation_errors.append(
                    f"Missing custom_id in result {position} of {len(results)}"
                )
            elif task_errors:
                failed_task_ids.add(result.custom_id)
            validation_errors.extend(task_errors)

        # A task with several lines (in one file, or in both the outputs and
        # the errors file) has no single result to trust
        lines_per_task = Counter(
            result.custom_id for result in batch_results if result.custom_id is not None
        )
        for custom_id, num_lines in lines_per_task.items():
            if num_lines > 1:
                failed_task_ids.add(custom_id)
                validation_errors.append(
                    f"Duplicate results for task {custom_id}: {num_lines} lines"
                )

        summary = self._count_task_results(failed_task_ids, batch_results)

        if validation_errors:
            raise ValueError(
                summary + "\n" + "Validation errors:\n" + "\n".join(validation_errors)
            )

        return True

    def _validate_result(self, result: BatchResult) -> List[str]:
        """Collect every validation error of one result.

        The common checks and the provider checks run independently, so a
        rejected prompt reports both its status code and its filter verdict.
        The checks return their findings and catch no exceptions: an
        exception here is a bug in a check, not bad data.

        Args:
            result: Single batch result

        Returns:
            List[str]: Validation error messages, empty if the result is valid
        """
        errors = self._validate_common_fields(result)
        # OpenAI has no provider-specific checks
        if self.config.is_endpoint_azure:
            errors.extend(self._validate_azure_specific(result))
        return errors

    @staticmethod
    def _count_task_results(
        failed_task_ids: set[str], results: List[BatchResult]
    ) -> str:
        """Count successful and failed tasks.

        Tasks are counted by their ``custom_id``, so a task with several
        result lines counts once. Error entries from the batch job are part
        of ``results`` and fail the common checks, so they are counted through
        ``failed_task_ids``. A result without ``custom_id`` cannot be matched
        to a task and counts as a failed one.

        Args:
            failed_task_ids: Identifiers of the tasks with validation errors
            results: List of batch results

        Returns:
            str: Summary of successful and failed tasks
        """
        task_ids = {
            result.custom_id for result in results if result.custom_id is not None
        }
        num_unidentified = sum(1 for result in results if result.custom_id is None)
        num_failed_tasks = len(failed_task_ids) + num_unidentified
        num_successful_tasks = len(task_ids - failed_task_ids)

        return (
            f"\nTask Summary:\n"
            f"Number of successful tasks: {num_successful_tasks}\n"
            f"Number of failed tasks: {num_failed_tasks}\n"
        )

    @staticmethod
    def _validate_common_fields(result: BatchResult) -> List[str]:
        """Validate fields common to both Azure and OpenAI responses.

        Args:
            result: Single batch result

        Returns:
            List[str]: Validation error messages, empty if the fields are valid
        """
        task = result.task
        errors: List[str] = []

        # Status code and error checks
        if result.status_code not in (None, HTTP_OK):
            errors.append(
                f"Failed request for task {task}: status code {result.status_code}"
            )
        if result.error_message is not None:
            errors.append(f"Error in task {task}: {result.error_message}")
        if errors:
            # A failed request has no choice to read
            return errors

        if result.status_code is None:
            return [f"Malformed result in task {task}: no status code"]
        if not result.choices:
            return [f"Malformed result in task {task}: no choices in the response body"]

        for choice in result.choices:
            finish_reason = choice.get("finish_reason", "stop")
            if finish_reason != "stop":
                errors.append(
                    f"Unexpected finish reason in task {task}: {finish_reason}"
                )
        return errors

    def _validate_azure_specific(self, result: BatchResult) -> List[str]:
        """Validate Azure-specific response fields.

        Args:
            result: Single batch result

        Returns:
            List[str]: Validation error messages, empty if the fields are valid
        """
        task = result.task

        # Handle error case with content filter results
        if result.has_error_filters:
            return self._validate_filter_block(result.error_filters, task, "error")

        # A failed request without filter details (e.g. a server error) has
        # nothing Azure-specific to inspect; the common checks report it.
        if result.request_failed:
            return []

        # Handle successful case. Azure returns one block per choice and one
        # per prompt entry, and leaves some of them null. An absent block is
        # read as a null one, and so is the block of an absent prompt entry.
        blocks = self._label_blocks(
            "completion",
            [choice.get("content_filter_results") for choice in result.choices],
        )
        blocks += self._label_blocks("prompt", result.prompt_filter_blocks or [None])

        # Check Azure content filters, each block on its own: a null
        # completion block must not hide the verdict on the prompt.
        errors: List[str] = []
        for source, filters in blocks:
            errors.extend(self._validate_filter_block(filters, task, source))
        return errors

    @staticmethod
    def _label_blocks(source: str, blocks: List[Any]) -> List[tuple[str, Any]]:
        """Name each content filter block for the messages.

        Args:
            source: Which blocks they are: "completion" or "prompt"
            blocks: Content filter blocks as returned by Azure

        Returns:
            List[tuple[str, Any]]: Each block with its name; the name carries
                the block's index when there is more than one
        """
        if len(blocks) == 1:
            return [(source, blocks[0])]
        return [(f"{source} {index}", block) for index, block in enumerate(blocks)]

    def _validate_filter_block(self, filters: Any, task: str, source: str) -> List[str]:
        """Validate one content filter block, which Azure can leave null.

        An empty block fails like a null one: neither carries a verdict, so
        the task cannot count as clean.

        Args:
            filters: Content filter block as returned by Azure, in any shape
            task: Task identifier
            source: Name of the block: "completion", "prompt" or "error",
                with its index when there are several

        Returns:
            List[str]: Validation error messages, empty if the block is valid
        """
        if filters is None or filters == {}:
            return [f"Missing {source} content filter results in task {task}"]
        if not isinstance(filters, dict):
            return [
                f"Malformed {source} content filter results in task {task}: "
                f"expected an object, got {type(filters).__name__}"
            ]
        return self._validate_content_filters(filters, task)

    def _validate_content_filters(
        self, filters: Dict[str, Any], task: str
    ) -> List[str]:
        """Validate content filter results.

        Args:
            filters: Content filter results dictionary
            task: Task identifier

        Returns:
            List[str]: One validation error message per failing filter
        """
        errors: List[str] = []
        for filter_name, filter_data in filters.items():
            error = self._validate_filter(filter_name, filter_data, task)
            if error is not None:
                errors.append(error)
        return errors

    @staticmethod
    def _validate_filter(
        filter_name: str, filter_data: Any, task: str
    ) -> Optional[str]:
        """Validate the result of one content filter.

        Args:
            filter_name: Name of the filter, e.g. "self_harm"
            filter_data: Result of the filter as returned by Azure, in any shape
            task: Task identifier

        Returns:
            Optional[str]: Validation error message, None if the filter passed
        """
        if not isinstance(filter_data, dict):
            return f"Missing {filter_name} filter result in task {task}"

        # Filters like jailbreak and protected_material report "detected",
        # the standard ones "filtered" and, except for blocklists, a severity.
        is_detection = filter_name == "jailbreak" or "detected" in filter_data
        verdict_field = "detected" if is_detection else "filtered"
        if verdict_field not in filter_data:
            return f"Missing required fields in task {task}: {verdict_field!r}"
        if is_detection:
            label = "Jailbreak attempt" if filter_name == "jailbreak" else filter_name
            return (
                f"{label} detected in task {task}" if filter_data["detected"] else None
            )
        if filter_data["filtered"]:
            return f"{filter_name} filter triggered in task {task}"
        if filter_data.get("severity", "safe") not in ACCEPTED_SEVERITIES:
            return f"{filter_name} has concerning severity in task {task}"
        return None
