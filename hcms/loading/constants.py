"""Constants used in data loading."""

from pathlib import Path

from hcms.constants import DATA_DIR

# Directories
TEST_DIR: Path = Path("tests")
FIXTURES_DIR: Path = Path("fixtures")
DATASET_DIR: Path = DATA_DIR / "dataset"

# Data paths
DATASET_2_CONVERSATION_STARTERS: Path = DATA_DIR / "dataset-2-conversation-starters.csv"
DATASET_3_CONVERSATION_STARTERS: Path = DATA_DIR / "dataset-3-conversation-starters.csv"

# Prompt paths
COMPANION_SYSTEM_PROMPT_PATH: Path = DATA_DIR / "companion-system-prompt.txt"
PSYCHIATRIST_SYSTEM_PROMPT_PATH: Path = DATA_DIR / "psychiatrist-system-prompt.txt"
RATE_CONVERSATIONS_PROMPT_PATH: Path = DATA_DIR / "rate-conversations-prompt.txt"
RATE_MESSAGES_PROMPT_PATH: Path = DATA_DIR / "rate-messages-prompt.txt"
