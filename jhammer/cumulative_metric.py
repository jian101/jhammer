from typing import Union

import torch

from jhammer.reductions import Reduction
from jhammer.type_converters import convert_2_data_type


class CumulativeMetric:
    def __init__(self, device=None):
        self.buffer = None
        self.device = device

    def __call__(self, value):
        if isinstance(value, Union[list, tuple, set]):
            self.extend(value)
        elif hasattr(value, "ravel"):
            value = value.ravel()
            self.extend(value)
        else:
            self.append(value)

    def append(self, value):
        if self.buffer is None:
            self.buffer = []
        value = convert_2_data_type(value, torch.Tensor, dtype=torch.float64, device=self.device)
        self.buffer.append(value)

    def extend(self, value):
        if self.buffer is None:
            self.buffer = []
        value = convert_2_data_type(value, torch.Tensor, dtype=torch.float64, device=self.device)
        self.buffer.extend(value)

    def aggregate(self, reduction="mean"):
        if self.buffer is None:
            return self.buffer
        match reduction:
            case Reduction.MEAN:
                return torch.stack(self.buffer).mean()
            case Reduction.SUM:
                return torch.stack(self.buffer).sum()
            case Reduction.NONE:
                return torch.stack(self.buffer)
            case _:
                raise ValueError(f"Unsupported reduction method {reduction}.")

    def get_buffer(self):
        return self.buffer

    def reset(self):
        self.buffer = None
