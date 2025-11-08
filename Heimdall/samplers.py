import math

import numpy as np
from torch.utils.data import BatchSampler, DistributedSampler, Subset

from Heimdall.datasets import PartitionedDataset, PartitionedSubset
from Heimdall.utils import AllPartitionsExhausted


class PartitionedDistributedSampler(DistributedSampler):
    """Distributed sampler for `PartitionedDataset`."""

    def __init__(self, dataset: PartitionedSubset | PartitionedDataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.unshuffled = True

        if isinstance(self.dataset, Subset):
            subset = dataset
            self.full_dataset = subset.dataset
            self.partition_sizes = {partition: len(indices) for partition, indices in subset._indices.items()}
        else:
            self.full_dataset = dataset
            self.partition_sizes = self.full_dataset._data.partition_sizes

        self.rng = np.random.default_rng(seed=self.full_dataset._data._cfg.seed)

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

        self.partition_order = list(range(self.num_partitions))
        self.partition_idx = None

        self.iterator = PartitionIndexIterator(self)

    @property
    def partition_idx(self):
        return self._partition_idx

    @property
    def num_partitions(self):
        return self.full_dataset.num_partitions

    @property
    def num_cells(self):
        return self.full_dataset.num_cells

    @partition_idx.setter
    def partition_idx(self, partition_idx: int | None):
        self._partition_idx = partition_idx
        if partition_idx is None:
            return

        partition = self.partition_order[partition_idx]

        # load underlying partition
        self.full_dataset.partition = partition

        self.iterator.set_indices(self.generate_partition_indices(partition))

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

        yield from indices

    # def __iter__(self):
    #     return self.generate_partition_indices(self.full_dataset.partition)

    def __iter__(self):
        if self.shuffle and self.unshuffled:
            self.partition_order = self.rng.permutation(self.partition_order)
            self.unshuffled = False

        if self.partition_idx is None:
            self.partition_idx = 0

        return self.iterator

    def __len__(self) -> int:
        return sum(self.total_samples_per_partition.values()) // self.num_replicas


class PartitionIndexIterator:
    def __init__(self, sampler: PartitionedDistributedSampler):
        self.sampler = sampler

    def set_indices(self, partition_indices: list):
        self.partition_indices = partition_indices

    def __next__(self):
        try:
            return next(self.partition_indices)
        except StopIteration as e:
            if self.sampler.partition_idx + 1 == self.sampler.num_partitions:
                self.sampler.unshuffled = True
                raise AllPartitionsExhausted()

            raise e


class PartitionedBatchSampler(BatchSampler):
    """Virtualized batched sampling from multiple partitions.

    Iterates through the indices provided by the PartitionedDistributedSampler for a partition until
    the end, and then moves onto the next partition and tries again. If no more partitions exist
    (as detected by an `AllPartitionsExhausted` exception), the epoch is finished.

    """

    def __iter__(self):
        while True:
            try:
                batch = []
                for idx in self.sampler:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

                if len(batch) > 0:  # flush remainder
                    yield batch

                self.sampler.partition_idx += 1

            except AllPartitionsExhausted:
                self.sampler.partition_idx = None
                if len(batch) > 0:  # flush remainder
                    yield batch
                break
