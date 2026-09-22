"""Constants used in monitoring."""

from enum import Enum, auto, unique


class RegTestColumns(str, Enum):
    """Column names in RegTestProcessor."""

    QUESTION = "question"
    ANSWER = "answer"


@unique
class BatchStatus(Enum):
    """Possible statuses for a batch job.

    The values are unique and automatically generated from the name of the enum member.

    Reference:
    https://docs.python.org/3/howto/enum.html#using-automatic-values
    """

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    VALIDATING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELED = auto()

    @classmethod
    def terminal_states(cls) -> set[str]:
        """States that indicate the batch job has finished."""
        return {cls.COMPLETED.value, cls.FAILED.value, cls.CANCELED.value}
