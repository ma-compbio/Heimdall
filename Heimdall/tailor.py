from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from Heimdall.fc import Fc


class Tailor(ABC):
    def __init__(
        self,
        fc: Fc,
    ):
        self.fc = fc

    @abstractmethod
    def pad(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Pad tokenization that is smaller than desired input length.

        Args:
            cell_tokenization: the stacked gene identity- and gene expression-based tokenization
                dof a cell.

        """

        (input_length,) = identity_inputs.shape
        padding_args = {
            "pad_width": ((0, self.fc.max_input_length - input_length)),
            "mode": "constant",
            "constant_values": (0, np.nan),
        }
        padded_identity_inputs = np.pad(
            identity_inputs.astype(self.fc.float_dtype),
            **padding_args,
        )

        padded_expression_inputs = np.pad(
            expression_inputs.astype(self.fc.float_dtype),
            **padding_args,
        )

        padded_identity_inputs[np.isnan(padded_identity_inputs).nonzero()] = self.fc.fg.pad_value
        padded_expression_inputs[np.isnan(padded_expression_inputs).nonzero()] = self.fc.fe.pad_value

        return padded_identity_inputs, padded_expression_inputs

    @abstractmethod
    def limit(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Limit tokenization that exceeds the desired input length.

        Args:
            cell_tokenization: the stacked gene identity- and gene expression-based tokenization
                of a cell.

        """
        identity_inputs = identity_inputs[: self.fc.max_input_length].astype(self.fc.float_dtype)
        expression_inputs = expression_inputs[: self.fc.max_input_length].astype(self.fc.float_dtype)

        return identity_inputs, expression_inputs

    def __call__(self, identity_inputs: NDArray, expression_inputs: NDArray, gene_order: NDArray) -> NDArray:
        (input_length,) = identity_inputs.shape

        if input_length >= self.fc.max_input_length:
            identity_inputs, expression_inputs = self.limit(identity_inputs, expression_inputs, gene_order)
            # print(f"{identity_inputs=}")
            # print(f"{expression_inputs=}")

        (input_length,) = identity_inputs.shape

        if input_length < self.fc.max_input_length:
            identity_inputs, expression_inputs = self.pad(identity_inputs, expression_inputs, gene_order)

        return identity_inputs, expression_inputs


class ReorderTailor(Tailor):
    def limit(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:

        identity_inputs = identity_inputs[gene_order]
        expression_inputs = expression_inputs[gene_order]

        return super().limit(identity_inputs, expression_inputs, gene_order)

    def pad(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:

        identity_inputs = identity_inputs[gene_order]
        expression_inputs = expression_inputs[gene_order]

        return super().pad(identity_inputs, expression_inputs, gene_order)


class ChromosomeTailor(Tailor):
    def __init__(self, fc: Fc, sample_size: int):
        self.sample_size = sample_size

        super().__init__(fc=fc)

    def weighted_resampling(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Weighted sampling."""

        weights = np.log1p(expression_inputs)
        weights /= np.sum(weights)

        resampled_indices = self.fc.rng.choice(
            len(identity_inputs),
            size=self.sample_size,
            p=weights,
            replace=True,
        )

        resampled_identity_inputs = identity_inputs[resampled_indices]
        resampled_expression_inputs = expression_inputs[resampled_indices]

        choosen_chrom = self.fc.chroms.iloc[resampled_identity_inputs]
        (input_length,) = resampled_identity_inputs.shape

        num_chromosomes = len(self.fc.shuffled_chromosomes)
        raw_sequence_length = input_length + 2 * num_chromosomes

        grouped_gene_tokenization = np.full(raw_sequence_length, self.fc.fg.pad_value)
        grouped_expression_tokenization = np.full(raw_sequence_length, self.fc.fe.pad_value)

        sequence_index = 0
        gene_ranks = np.argsort(gene_order)
        resampled_gene_ranks = gene_ranks[resampled_indices]

        for chromosome in self.fc.shuffled_chromosomes:
            (chromosome_index,) = np.where(choosen_chrom == chromosome)

            chromosome_identity_inputs = resampled_identity_inputs[chromosome_index]
            chromosome_expression_inputs = resampled_expression_inputs[chromosome_index]

            chromosome_gene_ranks = resampled_gene_ranks[chromosome_index]
            chromosome_gene_order = np.argsort(chromosome_gene_ranks)

            placeholder_id = -(chromosome + self.fc.chrom_token_offset + 1)

            grouped_gene_tokenization[sequence_index] = placeholder_id
            grouped_expression_tokenization[sequence_index] = placeholder_id
            # ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET
            # token of this chromosome # i = 1 next token is a chrom open

            sequence_index += 1
            # now sort the genes by start order within the chroms
            num_chromosome_genes = len(chromosome_index)

            chromosome_genes = chromosome_identity_inputs[chromosome_gene_order]
            chromosome_expression = chromosome_expression_inputs[chromosome_gene_order]

            grouped_gene_tokenization[sequence_index : (sequence_index + num_chromosome_genes)] = chromosome_genes
            grouped_expression_tokenization[sequence_index : (sequence_index + num_chromosome_genes)] = (
                chromosome_expression
            )

            sequence_index += num_chromosome_genes

            grouped_gene_tokenization[sequence_index] = -self.fc.chrom_token_offset
            grouped_expression_tokenization[sequence_index] = -self.fc.chrom_token_offset
            # ordered_choice_idx[i] = args.chrom_token_right_idx # add the chrom sep again

            sequence_index += 1  # add the closing token again

        return grouped_gene_tokenization, grouped_expression_tokenization

    def limit(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:

        return super().limit(identity_inputs, expression_inputs, gene_order)

    def pad(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:

        return super().pad(identity_inputs, expression_inputs, gene_order)

    def __call__(self, identity_inputs: NDArray, expression_inputs: NDArray, gene_order: NDArray) -> NDArray:
        identity_inputs, expression_inputs = self.weighted_resampling(identity_inputs, expression_inputs, gene_order)

        return super().__call__(identity_inputs, expression_inputs, gene_order)


class WeightedResampleTailor(Tailor):
    def __init__(self, fc: Fc, sample_size: int):
        self.sample_size = sample_size

        super().__init__(fc=fc)

    def weighted_resampling(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Weighted sampling."""

        weights = np.log1p(expression_inputs)
        weights /= np.sum(weights)

        resampled_indices = self.fc.rng.choice(
            len(identity_inputs),
            size=self.sample_size,
            p=weights,
            replace=True,
        )

        resampled_identity_inputs = identity_inputs[resampled_indices]
        resampled_expression_inputs = expression_inputs[resampled_indices]

        return resampled_identity_inputs, resampled_expression_inputs

    def limit(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:

        return super().limit(identity_inputs, expression_inputs, gene_order)

    def pad(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:

        return super().pad(identity_inputs, expression_inputs, gene_order)

    def __call__(self, identity_inputs: NDArray, expression_inputs: NDArray, gene_order: NDArray) -> NDArray:
        identity_inputs, expression_inputs = self.weighted_resampling(identity_inputs, expression_inputs, gene_order)

        return super().__call__(identity_inputs, expression_inputs, gene_order)
