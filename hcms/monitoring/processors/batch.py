"""Batch Processing with (Azure) OpenAI API"""

import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl
from hcms.constants import CONTENT, ROLE, SYSTEM, USER
from hcms.monitoring.api import BatchConfig, create_client
from hcms.monitoring.constants import BatchStatus
from openai.lib._parsing._completions import type_to_response_format_param
from openai.types.batch import Batch
from pydantic import BaseModel, ValidationError


class BatchProcessor:
    """Handles OpenAI batch processing operations.

    Args:
        config: BatchConfig instance
        client: OpenAI client instance
    """

    def __init__(
        self,
        config: BatchConfig,
    ):
        self.config: BatchConfig = config
        self.client = create_client()
        self._batch_endpoint: str = "/v1/chat/completions"
        self._completion_window: str = "24h"
        self.batch_job: Optional[Batch] = None

    # Create base request body
    def _create_request_body(
        self, idx: int, content: str, response_format_model: Optional[BaseModel]
    ) -> dict:
        """Creates the request body for batch processing.

        Args:
            idx: Index of the request
            content: Content to be processed
            response_format_model: Optional response format model
        Returns:
            dict: Request body
        """

        # Escape content string
        escaped_content = json.dumps(content)[1:-1]

        body = {
            "model": self.config.batch_model,
            "temperature": self.config.temperature,
            "messages": [
                {ROLE: SYSTEM, CONTENT: self.config.system_prompt},
                {ROLE: USER, CONTENT: escaped_content},
            ],
        }

        # Add response_format if model is provided
        if response_format_model:
            body["response_format"] = type_to_response_format_param(
                response_format_model
            )

        return {
            "custom_id": f"task-{idx}",
            "method": "POST",
            "url": self._batch_endpoint,
            "body": body,
        }

    def create_batch_file(
        self,
        df: pl.DataFrame,
        column_name: str,
        response_format_model: Optional[BaseModel],
    ) -> Path:
        """Creates JSONL batch file from DataFrame.

        Args:
            df: DataFrame
            column_name: Column name to use for batch processing

        Returns:
            Path: Path to batch file
        """
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        batch_file = self.config.output_dir / "batch_requests.jsonl"

        # Get data as columns
        indices = df.with_row_index().get_column("index").to_list()
        contents = df.get_column(column_name).to_list()

        # Create requests list using columnar data
        requests = [
            self._create_request_body(idx, content, response_format_model)
            for idx, content in zip(indices, contents)
        ]

        with open(batch_file, "w", encoding=self.config.encoding) as f:
            for request in requests:
                f.write(json.dumps(request) + "\n")

        return batch_file

    def submit_batch(self, file_name: Path) -> str:
        """Submits batch job to OpenAI.

        Args:
            file_name: Path to JSONL batch file

        Returns:
            str: Batch job ID
        """
        with open(file_name, "rb") as f:
            batch_file = self.client.files.create(file=f, purpose="batch")

        # Wait for file processing (undocumented)
        # pylint: disable-next=line-too-long
        # Source: https://github.com/openai/openai-python/blob/d9c966dea77fa3493114865a7f785f3134f1cc1e/src/openai/resources/files.py#L332-L353
        self.client.files.wait_for_processing(batch_file.id)

        batch_job = self.client.batches.create(
            input_file_id=batch_file.id,
            endpoint=self._batch_endpoint,
            completion_window=self._completion_window,
        )

        self.batch_job = batch_job

        return batch_job.id

    def _get_current_time(self) -> str:
        """Gets the current time formatted as a string.

        Returns:
            str: The current time in the format "%Y-%m-%d %H:%M:%S".
        """
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _get_batch_job_status(self, job_id: str) -> tuple[Batch, str]:
        """Retrieves batch job and returns its current status.

        Args:
            job_id: Batch job ID

        Returns:
            tuple[Batch, str]: Tuple containing the batch job and its status

        Raises:
            RuntimeError: If batch response is invalid
        """
        try:
            batch_job = Batch.model_validate(self.client.batches.retrieve(job_id))
            self.batch_job = batch_job
            status = batch_job.status
            return batch_job, status
        except ValidationError as e:
            raise RuntimeError(f"Invalid batch response: {e}") from e

    def _retrieve_batch_job(self, job_id: str) -> Batch:
        """Retrieves and monitors batch job status until completion.

        Args:
            job_id: Batch job ID

        Returns:
            Batch: Completed batch job

        Raises:
            RuntimeError: If batch response is invalid
        """
        batch_job, status = self._get_batch_job_status(job_id)
        self.batch_job = batch_job
        one_minute_in_seconds: int = 60

        while status not in BatchStatus.terminal_states():
            print(f"{self._get_current_time()} Waiting for batch job to finish...")
            time.sleep(one_minute_in_seconds)

            try:
                batch_job = Batch.model_validate(self.client.batches.retrieve(job_id))
                self.batch_job = batch_job
                status = batch_job.status
                print(
                    f"{self._get_current_time()} Batch Id: {job_id}, Status: {status}"
                )
            except ValidationError as e:
                raise RuntimeError(f"Invalid batch response: {e}") from e

        return batch_job

    def _validate_batch_job_completion(self, batch_job: Batch) -> None:
        """Validates batch job completion status.

        Args:
            batch_job: Completed batch job

        Raises:
            RuntimeError: If job failed, was canceled, or has no output file
        """
        self.batch_job = batch_job
        if batch_job.status == BatchStatus.FAILED.value:
            error_messages = [
                f"Error code {error.code} Message {error.message}"
                for error in batch_job.errors.data
            ]
            raise RuntimeError(f"Batch job failed: {error_messages}")

        if batch_job.status == BatchStatus.CANCELED.value:
            raise RuntimeError("Batch job was canceled")

        if batch_job.output_file_id is None:
            raise RuntimeError("Batch job has no output file")

    def _process_jsonl_file(
        self, content: bytes, file_prefix: str, job_id: str
    ) -> List[Dict[str, Any]]:
        """Process JSONL file content and return parsed results.

        Args:
            content: Raw file content
            file_prefix: Prefix for output filename
            job_id: Batch job ID

        Returns:
            List[Dict]: Parsed results
        """
        result_file = self.config.output_dir / f"{file_prefix}_{job_id}.jsonl"
        with open(result_file, "wb") as f:
            f.write(content)

        results = []
        with open(result_file, "r", encoding=self.config.encoding) as f:
            for line in f:
                results.append(json.loads(line.strip()))

        return results

    def retrieve_outputs(self, batch_job: Batch, job_id: str) -> List[Dict[str, Any]]:
        """Retrieves and processes batch job outputs.

        Args:
            batch_job: Completed batch job
            job_id: Batch job ID

        Returns:
            List[Dict]: Parsed outputs

        Raises:
            RuntimeError: If no outputs found in output
        """
        self.batch_job = batch_job
        outputs = self._process_jsonl_file(
            self.client.files.content(batch_job.output_file_id).content,
            "outputs",
            job_id,
        )

        return outputs

    def retrieve_errors(self, batch_job: Batch, job_id: str) -> List[Dict[str, Any]]:
        """Retrieves and processes batch job errors.

        Args:
            batch_job: Completed batch job
            job_id: Batch job ID
        Returns:
            List[Dict]: Parsed errors
        """
        self.batch_job = batch_job
        if batch_job.error_file_id is None:
            return []

        errors = self._process_jsonl_file(
            self.client.files.content(batch_job.error_file_id).content, "errors", job_id
        )

        return errors

    def get_results(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieves and saves job results.

        Results are a combination of (successful) outputs and errors.

        Args:
            job_id: Batch job ID

        Returns:
            List[Dict]: Parsed results

        Raises:
            RuntimeError: If the batch job fails
        """
        batch_job = self._retrieve_batch_job(job_id)
        self.batch_job = batch_job
        self._validate_batch_job_completion(batch_job)
        outputs: List[Dict[str, Any]] = self.retrieve_outputs(batch_job, job_id)

        errors = self.retrieve_errors(batch_job, job_id)
        outputs += errors

        return outputs
