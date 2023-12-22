from enum import StrEnum


class Reduction(StrEnum):
    """
    Reduction policies.
    """
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"
