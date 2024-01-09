import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries

from jhammer.type_conversion import convert_2_data_type


def distance_transform_sdf(input, normalize=False):
    """
    Compute signed distance function(SDF) of the input.

    Args:
        input: input data ndarray or tensor.
        normalize: if True, return the normalization of SDF which the values belong to [0,1]. Default is false.
    """
    binary_segmentation_np = convert_2_data_type(input, output_type=np.ndarray, dtype=bool)
    pos_distance = distance_transform_edt(binary_segmentation_np)
    neg_segmentation = ~binary_segmentation_np
    neg_distance = distance_transform_edt(neg_segmentation)

    boundary = find_boundaries(binary_segmentation_np, mode="inner")
    eps = 1e-6
    if normalize:
        sdf = (neg_distance - neg_distance.min()) / (neg_distance.max() - neg_distance.min() + eps) - \
              (pos_distance - pos_distance.min()) / (pos_distance.max() - pos_distance.min() + eps)
    else:
        sdf = neg_distance - pos_distance
    sdf[boundary] = 0

    sdf = convert_2_data_type(sdf, output_type=type(input), device=input.device if hasattr(input, "device") else None)
    return sdf
