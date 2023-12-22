import numpy as np
from einops import rearrange

from jhammer.distance_transforms import distance_transform_sdf


class ToType:
    def __init__(self, dtype, elements=None):
        if elements is None:
            elements = ["image", ]
        if isinstance(elements, str):
            elements = [elements, ]
        self.dtype = dtype
        self.elements = elements

    def __call__(self, data):
        for elem in self.elements:
            value = data[elem].astype(self.dtype)
            data[elem] = value
        return data


class Rearrange:
    """
    Change the arrangement of given elements.
    Args:
        pattern:
        elements:
    """

    def __init__(self, pattern, elements=None):
        if elements is None:
            elements = ["image", ]
        if isinstance(elements, str):
            elements = [elements, ]
        self.pattern = pattern
        self.elements = elements

    def __call__(self, data):
        for elem in self.elements:
            value = data[elem]
            value = rearrange(value, self.pattern)
            data[elem] = value
        return data


class AddChannel:
    def __init__(self, dim, elements=None):
        if elements is None:
            elements = ["image", ]
        if isinstance(elements, str):
            elements = [elements, ]
        self.dim = dim
        self.elements = elements

    def __call__(self, data):
        for elem in self.elements:
            value = data[elem]
            value = np.expand_dims(value, axis=self.dim)
            data[elem] = value
        return data


class MinMaxNormalization:
    """
    Perform min-max normalization.
    Args:
        lower_bound_percentile: intensity of `lower_bound_percentile` is the min value, default is 1.
        upper_bound_percentile: intensity of `upper_bound_percentile` is the max value, default is 99.
        elements: which elements of the data to perform the operation.
    """

    def __init__(self, lower_bound_percentile=1, upper_bound_percentile=99, elements=None):
        if elements is None:
            elements = ["image", ]
        if isinstance(elements, str):
            elements = [elements, ]
        self.lower_bound_percentile = lower_bound_percentile
        self.upper_bound_percentile = upper_bound_percentile
        self.elements = elements

    def __call__(self, data):
        for element in self.elements:
            image = data[element]
            min_value, max_value = np.percentile(image, (self.lower_bound_percentile, self.upper_bound_percentile))
            image = (image - min_value) / (max_value - min_value)
            data[element] = image
        return data


class SDF:
    def __init__(self, normalize=True, elements=None):
        """
        Compute signed distance function.
        Args:
            normalize: if `True`, normalize the SDF by min-max normalization.
            elements: binary segmentation for computing SDF
        """
        if isinstance(elements, str):
            elements = [elements, ]
        self.elements = elements
        self.normalize = normalize

    def __call__(self, data):
        for element in self.elements:
            segmentation = data[element]
            sdf = distance_transform_sdf(segmentation, self.normalize)
            data[f"SDF_{element}"] = sdf
        return data


class GetShape:
    def __init__(self, elements=None):
        if elements is None:
            elements = ["image", ]
        if isinstance(elements, str):
            elements = [elements, ]
        self.elements = elements

    def __call__(self, data):
        for elem in self.elements:
            shape = data[elem].shape
            shape = np.asarray(shape)
            data[f"{elem}_shape"] = shape
        return data
