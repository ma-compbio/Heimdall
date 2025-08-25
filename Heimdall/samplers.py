import math

import torch
from torch.utils.data import DistributedSampler


class PartitionedDistributedSampler(DistributedSampler):
    """Distributed sampler for `PartitionedDataset`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.total_samples_per_partition = {}

        for p, part_size in self.dataset._data.partition_sizes.items():
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

    def generate_partition_indices(self, partition, rand_generator=None):

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=rand_generator).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

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

        return indices[:500]  # TODO: why 500?

    def __iter__(self):
        partition_order = list(range(self.dataset.num_partitions))  # Provide an arg to shuffle
        g = None

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # partition_order = torch.randperm(partition_order, generator=rand_generator).tolist()
            # NOT SUPPORTED YET, WILL NEED TO MAKE SURE PARTITION IS LOADED

        for i, partition in enumerate(partition_order):

            partition_indices = self.generate_partition_indices(partition, rand_generator=g)

            yield from partition_indices

            # trigger loading of next partition
            if i < self.dataset.num_partitions - 1:
                self.dataset.partition = partition_order[i + 1]
            # trigger end of epoch
            else:
                raise StopIteration

    def __len__(self) -> int:
        return sum(self.num_samples_per_partition.values())
