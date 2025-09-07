import math

import numpy as np
from torch.utils.data import DistributedSampler, Subset

from Heimdall.datasets import PartitionedDataset, PartitionedSubset


class PartitionedDistributedSampler(DistributedSampler):
    """Distributed sampler for `PartitionedDataset`."""

    def __init__(self, dataset: PartitionedSubset | PartitionedDataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

        if isinstance(self.dataset, Subset):
            subset = dataset
            self.full_dataset = subset.dataset
            self.partition_sizes = {partition: len(indices) for partition, indices in subset.indices.items()}
        else:
            self.full_dataset = dataset
            self.partition_sizes = self.full_dataset._data.partition_sizes

        self.rng = np.random.default_rng(seed=self.full_dataset._data._cfg.seed)
        self.partition_order = list(range(self.full_dataset.num_partitions))  # Provide an arg to shuffle

        self.total_samples_per_partition = {}

        for p, part_size in self.partition_sizes.items():
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and part_size % self.num_replicas != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                num_samples_part = math.ceil(
                    (part_size - self.num_replicas) / self.num_replicas,  # type: ignore[arg-type]
                )
            else:
                num_samples_part = math.ceil(part_size / self.num_replicas)  # type: ignore[arg-type]

            self.total_samples_per_partition[p] = num_samples_part * self.num_replicas

    def generate_partition_indices(self, partition):
        indices = list(range(self.partition_sizes[partition]))
        if self.shuffle:
            indices = self.rng.permutation(indices).astype(int).tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_samples_per_partition[partition] - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_samples_per_partition[partition]]
        assert len(indices) == self.total_samples_per_partition[partition]

        # subsample
        indices = indices[self.rank : self.total_samples_per_partition[partition] : self.num_replicas]
        assert len(indices) == self.total_samples_per_partition[partition] / self.num_replicas

        return iter(indices)  # TODO: why 500?

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.partition_order)

        for partition in self.partition_order:
            self.full_dataset.partition = partition
            indices = self.generate_partition_indices(partition)
            yield from indices

    def __len__(self) -> int:
        return sum(self.total_samples_per_partition.values()) // self.num_replicas
