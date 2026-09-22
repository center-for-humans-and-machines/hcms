"""Functions that are needed for calling moderation functions that will test the safety of the
dataset."""

from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
import polars as pl
from hcms.loading.constants import DATASET_DIR
from hcms.loading.data_loading import _load_json_with_config
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm


def convert_json_to_csv(
    input_dir: Path,
    output_dir: Path,
    file_pattern: str = "rated_dataset-round-2-*.json",
):
    """Converts multiple JSON files to CSV format.

    This function reads all JSON files matching the given pattern in the input
    directory, processes them for CSV compatibility, and writes them as CSV
    files to the output directory.

    Processing includes:
    - Replacing newline characters in the 'message' column.
    - Converting list and struct columns to string representations.

    Args:
        input_dir: The directory containing the JSON files.
        output_dir: The directory where the CSV files will be saved.
        file_pattern: A glob pattern to select the JSON files to convert.

    Raises:
        FileNotFoundError: If no files are found matching the pattern.
    """
    output_dir.mkdir(exist_ok=True)

    # Get list of matching files first to check if any exist
    matching_files = list(input_dir.glob(file_pattern))

    if not matching_files:
        raise FileNotFoundError(
            f"No files found in {input_dir} matching pattern '{file_pattern}'"
        )

    for model_file in matching_files:
        print(f"Converting {model_file.name} to CSV...")
        df = pl.read_json(model_file)

        # Prepare message column for CSV
        df = df.with_columns(
            pl.col("message").cast(pl.Utf8).str.replace_all(r"(\r\n|\r|\n)+", r"\\n")
        )

        # Convert complex types to strings for CSV compatibility
        for col_name in df.columns:
            if df[col_name].dtype == pl.List:
                df = df.with_columns(
                    pl.col(col_name).map_elements(
                        # pylint: disable-next=unnecessary-lambda
                        lambda x: str(x),
                        return_dtype=pl.Utf8,
                    )
                )
            elif df[col_name].dtype == pl.Struct:
                df = df.with_columns(pl.col(col_name).cast(pl.Utf8))

        # Write to CSV in the output directory
        output_path = output_dir / model_file.with_suffix(".csv").name
        df.write_csv(
            output_path,
            separator=",",
            quote_style="always",
            quote_char='"',
            line_terminator="\n",
            include_header=True,
        )
        print(f"Successfully saved {output_path}")


def reformat_df(filename: str) -> pl.DataFrame:
    """Expands a conversation DataFrame into a message-level DataFrame.

    This function reads a JSON file containing conversation data, where each
    record represents an entire conversation with multiple messages. It then
    transforms this data into a "long" format, where each row corresponds to a
    single message from a conversation, filtering for assistant messages only.

    This transformation is useful for tasks that require analysis or processing
    of individual assistant messages, such as content moderation testing.

    Args:
        filename: The name of the JSON file within the `DATASET_DIR`
            to be processed.

    Returns:
        A new DataFrame where each row represents a single
        assistant message, containing the columns: `conversation_id`, `turn`,
        `role`, and `message`.
    """
    df = _load_json_with_config(DATASET_DIR / filename)
    rows = []
    for i, conversation in enumerate(df["conversation"]):
        for j, message in enumerate(conversation):
            if message["role"] == "assistant":
                # Collect rows in a list of dicts
                rows.append(
                    {
                        # pylint: disable=unsubscriptable-object
                        "conversation_id": df["conversation_id"][i],
                        "turn": j,
                        "role": message["role"],
                        "message": message["content"],
                    }
                )
        # After the loop, create a new DataFrame from rows
    return pl.DataFrame(rows)


# pylint: disable-next=too-few-public-methods
class RatingConfig(BaseModel):
    """Configuration for rating and saving conversations."""

    file_pattern: str
    rating_function: Callable
    output_column_name: str
    output_column_dtype: Union[pl.DataType, Any]
    input_dir: Path = DATASET_DIR
    output_dir: Path = DATASET_DIR
    output_prefix: str = "rated-"

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _rate_and_save_conversations(config: RatingConfig) -> pl.DataFrame:
    """
    Processes dataset files, rates messages, and saves the results.

    Args:
        config: Configuration object containing all rating parameters.

    Returns:
        pl.DataFrame: The last rated DataFrame.

    Raises:
        FileNotFoundError: If no files are found matching the pattern.
    """
    config.output_dir.mkdir(exist_ok=True)

    # Get list of matching files first to check if any exist
    matching_files = list(config.input_dir.glob(config.file_pattern))

    if not matching_files:
        raise FileNotFoundError(
            f"No files found in {config.input_dir} matching pattern '{config.file_pattern}'"
        )

    rated_df = pl.DataFrame()  # Initialize empty DataFrame

    for model_file in matching_files:
        print(f"Processing {model_file.name}...")
        flattened_df = reformat_df(model_file.name)

        messages = flattened_df["message"].to_list()
        scores = [
            config.rating_function(message)
            for message in tqdm(messages, desc=f"Rating messages in {model_file.name}")
        ]

        rated_df = flattened_df.with_columns(
            pl.Series(
                config.output_column_name,
                scores,
                dtype=config.output_column_dtype,
                strict=False,
            ),
            pl.col("message").str.replace_all("\n", "\\n"),
        )

        output_filename = f"{config.output_prefix}{model_file.name}"
        output_path = config.output_dir / output_filename
        rated_df.write_json(output_path)
        print(f"Successfully wrote rated file to {output_path}")
    return rated_df


def create_excel_from_csvs(
    csv_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    file_pattern: str = "*.csv",
) -> None:
    """Creates an Excel file with multiple sheets from CSV files.

    This function reads all CSV files matching the given pattern in the specified
    directory and combines them into a single Excel file, with each CSV becoming
    a separate sheet.

    Args:
        csv_dir: The directory containing the CSV files. Defaults to DATASET_DIR/csv_format.
        output_file: The path for the output Excel file. Defaults to csv_dir/round2_rated.xlsx.
        file_pattern: A glob pattern to select the CSV files. Defaults to "*.csv".

    Raises:
        FileNotFoundError: If no CSV files are found matching the pattern.
    """
    if csv_dir is None:
        csv_dir = Path(DATASET_DIR) / "csv_format"

    if output_file is None:
        output_file = csv_dir / "round2_rated.xlsx"

    csv_files = list(csv_dir.glob(file_pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {csv_dir} matching pattern {file_pattern}"
        )

    # Create a new Excel writer
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Iterate through all CSV files in the folder
        for csv_file in csv_files:
            # Read each CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            # Use the file name (without extension) as the sheet name
            sheet_name = (
                csv_file.stem.split("round-2-")[1]
                if "round-2-" in csv_file.stem
                else csv_file.stem
            )
            print(f"Adding sheet: {sheet_name}")
            # Write the DataFrame to a sheet in the Excel file
            df.to_excel(
                writer, index=False, sheet_name=sheet_name[:31]
            )  # Excel sheet names max 31 chars

    print(f"Combined spreadsheet saved to: {output_file}")
