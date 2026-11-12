"""Base class for processing data using batch processing."""

from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import polars as pl
from pydantic import BaseModel

from hcms.constants import CONTENT, ROLE, SYSTEM, USER
from hcms.monitoring.api import BatchConfig
from hcms.monitoring.processors.batch import BatchProcessor


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

        for result in results:
            task_errors = self._validate_result(result)
            if task_errors:
                failed_task_ids.add(result["custom_id"])
                validation_errors.extend(task_errors)

        summary = self._count_task_results(failed_task_ids, results)

        if validation_errors:
            raise ValueError(
                summary + "\n" + "Validation errors:\n" + "\n".join(validation_errors)
            )

        return True

    def _validate_result(self, result: Dict[str, Any]) -> List[str]:
        """Collect every validation error of one result.

        The common checks and the provider checks run independently, so a
        rejected prompt reports both its status code and its filter verdict.
        A result whose shape a check cannot read is reported as malformed, so
        one bad line does not abort the report on all the others.

        Args:
            result: Single result dictionary

        Returns:
            List[str]: Validation error messages, empty if the result is valid
        """
        errors: List[str] = []
        for validate in (self._validate_common_fields, self._validate_provider):
            try:
                validate(result)
            except ValueError as e:
                errors.append(str(e))
            except (KeyError, IndexError, TypeError, AttributeError) as e:
                malformed = (
                    f"Malformed result in task {result['custom_id']}: "
                    f"{type(e).__name__}: {e}"
                )
                # Both checks read the same fields and can fail the same way
                if malformed not in errors:
                    errors.append(malformed)
        return errors

    def _validate_provider(self, result: Dict[str, Any]) -> None:
        """Run the provider-specific checks; OpenAI has none.

        Args:
            result: Single result dictionary

        Raises:
            ValueError: If validation fails
        """
        if self.config.is_endpoint_azure:
            self._validate_azure_specific(result)

    @staticmethod
    def _count_task_results(
        failed_task_ids: set[str], results: List[Dict[str, Any]]
    ) -> str:
        """Count successful and failed tasks.

        Error entries from the batch job are part of ``results`` and fail the
        common checks, so they are counted once, through ``failed_task_ids``.

        Args:
            failed_task_ids: Identifiers of the results with validation errors
            results: List of processing results

        Returns:
            str: Summary of successful and failed tasks
        """
        num_failed_tasks = len(failed_task_ids)
        num_successful_tasks = sum(
            1 for result in results if result["custom_id"] not in failed_task_ids
        )

        return (
            f"\nTask Summary:\n"
            f"Number of successful tasks: {num_successful_tasks}\n"
            f"Number of failed tasks: {num_failed_tasks}\n"
        )

    def _get_first_choice(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Get first choice from result or return error details.

        Args:
            result: Result dictionary

        Returns:
            Dict[str, Any]: First choice or error details
        """
        try:
            return result["response"]["body"]["choices"][0]
        except (KeyError, TypeError):
            return result["error"]["message"]["error"]

    def _get_message_content(self, result: Dict[str, Any]) -> str:
        """Get message content from result.

        Args:
            result: Result dictionary

        Returns:
            str: Message content
        """
        try:
            return self._get_first_choice(result)["message"]["content"]
        except (KeyError, TypeError):
            return result["error"]["message"]["error"]

    def _validate_common_fields(self, result: Dict[str, Any]) -> None:
        """Validate fields common to both Azure and OpenAI responses.

        Args:
            result: Single result dictionary

        Raises:
            ValueError: If validation fails
        """
        errors = []

        # Status code and error checks
        if result["response"]["status_code"] != 200:
            errors.append(
                f"Failed request for task {result['custom_id']}: "
                f"status code {result['response']['status_code']}"
            )

        if result["error"] is not None:
            error_message = result["error"]["message"]["error"]["message"]
            errors.append(f"Error in task {result['custom_id']}: {error_message}")

        # Only check finish reason if we have a successful response
        try:
            first_choice = self._get_first_choice(result)
            if (
                "finish_reason" in first_choice
                and first_choice["finish_reason"] != "stop"
            ):
                errors.append(
                    f"Unexpected finish reason in task {result['custom_id']}: "
                    f"{first_choice['finish_reason']}"
                )
        except (KeyError, IndexError, TypeError):
            # A failed request has no choice to read and is reported above.
            # Without such a report the result is malformed; the caller says so.
            if not errors:
                raise

        if errors:
            raise ValueError("\n".join(errors))

    def _validate_azure_specific(self, result: Dict[str, Any]) -> None:
        """Validate Azure-specific response fields.

        Args:
            result: Single result dictionary

        Raises:
            ValueError: If validation fails
        """
        custom_id = result["custom_id"]
        try:
            first_choice = self._get_first_choice(result)

            # Handle error case with content filter results
            if "innererror" in first_choice:
                self._validate_content_filters(
                    self._require_content_filters(
                        first_choice["innererror"]["content_filter_result"],
                        custom_id,
                        "error",
                    ),
                    custom_id,
                )
                return

            # A failed request without filter details (e.g. a server error) has
            # nothing Azure-specific to inspect; the common checks report it.
            if result["error"] is not None:
                return

            # Handle successful case. Azure returns some responses with a null
            # block on the choice or the prompt, so each is checked before use.
            choice_filters = self._require_content_filters(
                first_choice["content_filter_results"], custom_id, "completion"
            )
            prompt_filter_results = result["response"]["body"]["prompt_filter_results"]
            first_prompt = prompt_filter_results[0] if prompt_filter_results else None
            prompt_filters = self._require_content_filters(
                first_prompt["content_filter_results"] if first_prompt else None,
                custom_id,
                "prompt",
            )

            # Check Azure content filters
            self._validate_content_filters(choice_filters, custom_id)
            self._validate_content_filters(prompt_filters, custom_id)
        except (KeyError, IndexError) as e:
            raise ValueError(f"Missing required fields in task {custom_id}: {e}") from e

    @staticmethod
    def _require_content_filters(
        filters: Optional[Dict[str, Any]], custom_id: str, source: str
    ) -> Dict[str, Any]:
        """Return a content filter block, or fail when Azure left it null.

        An empty block is accepted: it lists no filters to check.

        Args:
            filters: Content filter block as returned by Azure, possibly null
            custom_id: Task identifier
            source: Which block it is: "completion", "prompt" or "error"

        Returns:
            Dict[str, Any]: The content filter block

        Raises:
            ValueError: If the block is missing or null
        """
        if filters is None:
            raise ValueError(
                f"Missing {source} content filter results in task {custom_id}"
            )
        return filters

    def _validate_content_filters(
        self, filters: Dict[str, Any], custom_id: str
    ) -> None:
        """Validate content filter results.

        Args:
            filters: Content filter results dictionary
            custom_id: Task identifier

        Raises:
            ValueError: If validation fails
        """
        for filter_name, filter_data in filters.items():
            if not isinstance(filter_data, dict):
                raise ValueError(
                    f"Missing {filter_name} filter result in task {custom_id}"
                )
            if filter_name == "jailbreak":
                if filter_data["detected"]:
                    raise ValueError(f"Jailbreak attempt detected in task {custom_id}")
            elif "detected" in filter_data:  # For filters like protected_material
                if filter_data["detected"]:
                    raise ValueError(f"{filter_name} detected in task {custom_id}")
            else:  # For standard filters
                if filter_data["filtered"]:
                    raise ValueError(
                        f"{filter_name} filter triggered in task {custom_id}"
                    )
                if filter_data["severity"] not in ["safe", "low"]:
                    raise ValueError(
                        f"{filter_name} has concerning severity in task {custom_id}"
                    )
