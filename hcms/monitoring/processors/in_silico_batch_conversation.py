"""Regression test processor for OpenAI Batch Processing API."""

from pathlib import Path

import polars as pl
from hcms.loading import COMPANION_SYSTEM_PROMPT_PATH, PSYCHIATRIST_SYSTEM_PROMPT_PATH
from hcms.monitoring.api import BatchConfig
from hcms.monitoring.processors.base import BaseProcessor
from hcms.utils import UTF_8, clean_text


class InSilicoConversationBatchProcessor(BaseProcessor):
    """Processor for synthetic conversation data using OpenAI Batch Processing API.

    Input: JSONL file with questions
    Output: JSONL file with answers
    """

    def __init__(self, input_file: str, clinician_prompts: bool = False):
        # Clinician prompts
        if clinician_prompts:
            prompt_path = PSYCHIATRIST_SYSTEM_PROMPT_PATH
        # Bot replies
        else:
            prompt_path = COMPANION_SYSTEM_PROMPT_PATH

        config = BatchConfig(
            system_prompt=clean_text(Path(prompt_path).read_text(encoding=UTF_8)),
            input_file=input_file,
            data_loader=pl.read_json,
        )
        super().__init__(
            config,
            input_column_name="Question",
            output_column_name="Answer",
        )
