from typing import Sequence

import numpy as np
import torch


def convert_2_dtype(data, dst_dtype):
    """
    Convert input data's data type to destination data type. If source data is a collection, convert the data type of
    elements.

    Args:
        data (object):
        dst_dtype (data type object):
    """

    if isinstance(data, np.ndarray):
        return data.astype(dst_dtype)
    if isinstance(data, torch.Tensor):
        return data.to(dst_dtype)
    type_data = type(data)
    if type_data in [list, tuple, set]:
        result = [dst_dtype(e) for e in data]
        return type_data(result)
    return dst_dtype(data)


def convert_2_data_type(data, output_type, dtype=None, device=None):
    """
    Convert data to the assigned type data. Note that the `dtype` is not functioned if the `output_type` is not in
    `[numpy.ndarray, torch.Tensor]`.

    Args:
        data (object):
        output_type (data type object):
        dtype (np.dtype or torch.dtype or None, optional, default=None):
        device (str or torch.device or None, optional, default=None):

    Returns:

    """

    if isinstance(data, output_type):
        if dtype:
            return convert_2_dtype(data, dtype)
        return data
    elif output_type == torch.Tensor:
        data = torch.as_tensor(data, dtype=dtype, device=device)
        return data
    elif output_type == np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            if dtype:
                data = data.astype(dtype)
        else:
            data = np.asarray(data, dtype)
        return data
    else:
        if hasattr(data, "tolist"):
            data = data.tolist
        else:
            data = list(data)
        data = _convert_data_recursively(data, output_type=output_type, dtype=dtype)
        return data if output_type == list else output_type(data)


def _convert_data_recursively(data: list, output_type=None, dtype=None):
    """
    Recursively change a subscriptable object's element type and data type.

    Args:
        data (list):
        output_type (data type object, optional, default=None):
        dtype (np.dtype or torch.dtype or None, optional, default=None):

    Returns:

    """

    if len(data) == 0:
        return output_type()
    if not isinstance(data[0], Sequence):
        if dtype is not None:
            if not isinstance(data[0], dtype):
                data = [dtype(e) for e in data]
    else:
        for i, e in enumerate(data):
            data[i] = _convert_data_recursively(e, output_type, dtype)

    if isinstance(data, Sequence):
        if output_type is not None and not isinstance(data, output_type):
            data = output_type(data)
    return data
