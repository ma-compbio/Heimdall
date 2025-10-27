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


class ChromosomeBlockTailor(Tailor):
    """
    Chromosome grouping without any resampling.
    - Groups genes into blocks per chromosome with open/close tokens.
    - Within each chromosome, genes are ordered according to `gene_order`.
    - Uses Tailor.limit (default truncation) and Tailor.pad.
    """
    def __init__(self, fc: Fc):
        super().__init__(fc=fc)

    def _group_by_chromosomes(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:
        # Number of original tokens (no sampling)
        (input_length,) = identity_inputs.shape

        # Pre-compute ranks (so lower rank = earlier)
        # Matches your pattern: np.argsort(gene_order) -> ranks used to sort within chrom
        gene_ranks = np.argsort(gene_order)

        # Map each gene id to its chromosome
        # Assumes `self.fc.chroms` is a pandas Series indexed by gene id
        chrom_of_gene = self.fc.chroms.iloc[identity_inputs]

        num_chromosomes = len(self.fc.shuffled_chromosomes)
        raw_sequence_length = input_length + 2 * num_chromosomes  # open + close for each chrom

        grouped_gene_tokenization = np.full(
            raw_sequence_length, self.fc.fg.pad_value, dtype=self.fc.float_dtype,
        )
        grouped_expression_tokenization = np.full(
            raw_sequence_length, self.fc.fe.pad_value, dtype=self.fc.float_dtype,
        )

        sequence_index = 0

        for chromosome in self.fc.shuffled_chromosomes:
            # indices of genes belonging to this chromosome
            chrom_idx = np.where(chrom_of_gene == chromosome)[0]

            # open token for this chromosome
            placeholder_id = -(chromosome + self.fc.chrom_token_offset + 1)
            grouped_gene_tokenization[sequence_index] = placeholder_id
            grouped_expression_tokenization[sequence_index] = placeholder_id
            sequence_index += 1

            # order genes within the chromosome using ranks derived from gene_order
            if chrom_idx.size > 0:
                chrom_gene_ranks = gene_ranks[chrom_idx]
                chrom_order = np.argsort(chrom_gene_ranks)

                chrom_genes = identity_inputs[chrom_idx][chrom_order]
                chrom_exprs = expression_inputs[chrom_idx][chrom_order]

                n = chrom_genes.shape[0]
                grouped_gene_tokenization[sequence_index : sequence_index + n] = chrom_genes
                grouped_expression_tokenization[sequence_index : sequence_index + n] = chrom_exprs
                sequence_index += n

            # close token
            grouped_gene_tokenization[sequence_index] = -self.fc.chrom_token_offset
            grouped_expression_tokenization[sequence_index] = -self.fc.chrom_token_offset
            sequence_index += 1

        # (sequence_index should equal raw_sequence_length)
        return grouped_gene_tokenization, grouped_expression_tokenization

    # Use default truncation (like ReorderTailor)
    def limit(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:
        return super().limit(identity_inputs, expression_inputs, gene_order)

    # Use default padding (like ReorderTailor)
    def pad(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:
        return super().pad(identity_inputs, expression_inputs, gene_order)

    def __call__(
        self,
        identity_inputs: NDArray,
        expression_inputs: NDArray,
        gene_order: NDArray,
    ) -> tuple[NDArray, NDArray]:
        # Build chromosome-blocked sequence (no resampling)
        identity_inputs, expression_inputs = self._group_by_chromosomes(
            identity_inputs, expression_inputs, gene_order,
            )
        # Then apply the standard Tailor length control (truncate/pad)
        return super().__call__(identity_inputs, expression_inputs, gene_order)
