"""Base class for processing data using batch processing."""

from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import polars as pl
from hcms.constants import CONTENT, ROLE, SYSTEM, USER
from hcms.monitoring.api import BatchConfig
from hcms.monitoring.processors.batch import BatchProcessor
from pydantic import BaseModel


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
            ValueError: If any result fails validation or row count mismatch
        """
        validation_errors = []

        # Validate row count
        if len(results) != self._num_rows:
            validation_errors.append(
                f"Number of results ({len(results)}) does not match "
                f"number of inputs to batch ({self._num_rows})\n"
            )

        for result in results:
            try:
                # Common validations
                self._validate_common_fields(result)

                # Provider-specific validations
                if self.config.is_endpoint_azure:
                    self._validate_azure_specific(result)
                elif self.config.is_endpoint_openai:
                    pass
                else:
                    validation_errors.append(
                        "Invalid endpoint type. Must be either Azure or OpenAI."
                    )
            except ValueError as e:
                validation_errors.append(str(e))

        # Count number of errors and report total number of successful and failed tasks
        summary = self._count_task_results(validation_errors, results)

        if validation_errors:
            raise ValueError(
                summary + "\n" + "Validation errors:\n" + "\n".join(validation_errors)
            )

        return True

    def _count_task_results(
        self, validation_errors: List[str], results: List[Dict[str, Any]]
    ) -> str:
        """Count successful and failed tasks.

        Args:
            validation_errors: List of validation error messages
            results: List of processing results

        Returns:
            str: Summary of successful and failed tasks
        """
        task_ids = set()
        for error in validation_errors:
            if "task" in error:
                task_id = error.split("task")[1].split(":")[0].strip()
                task_ids.add(task_id)

        batch_job = self._batch_processor.batch_job
        if batch_job:
            errors = self._batch_processor.retrieve_errors(
                batch_job=batch_job, job_id=batch_job.id
            )
        else:
            errors = []
        num_failed_tasks = len(task_ids) + len(errors)
        num_successful_tasks = len(results) - len(task_ids)

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
        except KeyError:
            pass  # Error case already handled above

        if errors:
            raise ValueError("\n".join(errors))

    def _validate_azure_specific(self, result: Dict[str, Any]) -> None:
        """Validate Azure-specific response fields.

        Args:
            result: Single result dictionary

        Raises:
            ValueError: If validation fails
        """
        try:
            first_choice = self._get_first_choice(result)

            # Handle error case with content filter results
            if "innererror" in first_choice:
                content_filter_result = first_choice["innererror"][
                    "content_filter_result"
                ]
                self._validate_content_filters(
                    content_filter_result, result["custom_id"]
                )
                return

            # Handle successful case
            choice_filters = first_choice["content_filter_results"]
            prompt_filters = result["response"]["body"]["prompt_filter_results"][0][
                "content_filter_results"
            ]

            # Check Azure content filters
            self._validate_content_filters(choice_filters, result["custom_id"])
            self._validate_content_filters(prompt_filters, result["custom_id"])
        except KeyError as e:
            raise ValueError(
                f"Missing required fields in task {result['custom_id']}: {e}"
            ) from e

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
