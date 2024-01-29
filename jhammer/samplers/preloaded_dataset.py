import random
from abc import ABCMeta, abstractmethod
from itertools import cycle

from jhammer.samplers.coordinate_generator import BalancedCoordinateGenerator
from jhammer.samplers.patch_picker import WeightedPatchPicker


class PreloadedDataset(metaclass=ABCMeta):
    """
    Store the subjects on a queue of length queue_length in the RAM.
    For 3d data training.
    """

    def __init__(self, samples: list,
                 patch_size,
                 n_patches_per_sample,
                 n_samples_alive=0,
                 shuffle=True,
                 transforms=None):
        """
        Args:
            samples: sample index.
            patch_size: size of each patch.
            n_patches_per_sample: how many patches cropped from every sample.
            n_samples_alive: maximum number of samples that save in RAM.
            shuffle: if ``True``, shuffle the samples.
            transforms:


        """
        if shuffle:
            random.shuffle(samples)
        self.sample_iterator = cycle(samples)

        self.patch_size = patch_size
        self.n_patches_per_sample = n_patches_per_sample
        self.n_samples_alive = n_samples_alive if 0 < n_samples_alive <= len(samples) else len(samples)

        self.sample_queue = [None] * self.n_samples_alive
        self.index = 0

        self.transforms = transforms

    def update_queue(self, index):
        """
        Update the queue element in {index}
        """
        sample_idx = next(self.sample_iterator)

        if self.sample_queue[index] and self.sample_queue[index][0] == sample_idx:
            # Just update the patch extractor.
            _, patch_picker = self.sample_queue[index]
            patch_picker.reset()
        else:
            data = self.get_sample(sample_idx)
            coordinate_generator = BalancedCoordinateGenerator(self.n_patches_per_sample,
                                                               self.get_weight_map(data),
                                                               self.patch_size)

            patch_picker = WeightedPatchPicker(data, self.patch_size, coordinate_generator)
            self.sample_queue[index] = (sample_idx, patch_picker)

    @abstractmethod
    def get_sample(self, index):
        """
        get sample by index from ``self.sample_iterator``
        Args:
            index:

        Returns:

        """
        ...

    @abstractmethod
    def get_weight_map(self, data):
        ...

    def __iter__(self):
        return self

    def __next__(self):
        index = self.index
        self.index = (self.index + 1) % self.n_samples_alive

        # Only invoke when visiting the index at the first time.
        if self.sample_queue[index] is None:
            self.update_queue(index)
        try:
            data = self.next(index)
        except StopIteration:
            self.update_queue(index)
            data = self.next(index)
        return data

    def next(self, index):
        _, patch_picker = self.sample_queue[index]
        data = next(patch_picker)
        if self.transforms:
            data = self.transforms(data)
        return data
