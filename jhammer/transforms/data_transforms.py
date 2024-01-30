import numpy as np
from einops import rearrange

from jhammer.transforms.transforms import Transform


class ToType(Transform):
    def __init__(self, keys, dtype):
        super().__init__(keys)
        self.dtype = dtype

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            value = data[key].astype(self.dtype)
            data[key] = value
        return data


class Rearrange(Transform):
    def __init__(self, keys, pattern):
        """
        Change the arrangement of given elements.

        Args:
            keys (str or sequence):
            pattern (str): Arranging pattern. For example "i j k -> j k i".
        """

        super().__init__(keys)
        self.pattern = pattern

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            value = data[key]
            value = rearrange(value, self.pattern)
            data[key] = value
        return data


class AddChannel(Transform):
    def __init__(self, keys, dim):
        """
        Add additional dimension in specific position.

        Args:
            keys (str or sequence):
            dim (int):
        """

        super().__init__(keys)
        self.dim = dim

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            value = data[key]
            value = np.expand_dims(value, axis=self.dim)
            data[key] = value
        return data


class MinMaxNormalization(Transform):
    def __init__(self, keys, lower_bound_percentile=1, upper_bound_percentile=99):
        """
        Perform min-max normalization.

        Args:
            lower_bound_percentile (int, optional, default=1):
            upper_bound_percentile (int, optional, default=99):
            keys (str or sequence):
        """

        super().__init__(keys)
        self.lower_bound_percentile = lower_bound_percentile
        self.upper_bound_percentile = upper_bound_percentile

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            image = data[key]
            min_value, max_value = np.percentile(image, (self.lower_bound_percentile, self.upper_bound_percentile))
            image = (image - min_value) / (max_value - min_value)
            data[key] = image
        return data


class GetShape(Transform):
    def __init__(self, keys):
        """
        Get array shape.

        Args:
            keys (str or sequence):
        """

        super().__init__(keys)

    def _call_fun(self, data, *args, **kwargs):
        for key in self.keys:
            shape = data[key].shape
            shape = np.asarray(shape)
            data[f"{key}_shape"] = shape
        return data
