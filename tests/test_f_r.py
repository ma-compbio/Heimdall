import torch
from pytest import fixture
from torch.nn import Embedding

from Heimdall.reduce import IdentityReduce


@fixture
def mock_gene_embeddings():
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
    )


@fixture
def mock_expression_embeddings():
    return torch.tensor(
        [
            [2, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 2],
        ],
        dtype=torch.float32,
    )


@fixture
def identity_reduce(geneformer_fc):
    reducer = IdentityReduce(geneformer_fc)
    return reducer


def test_identity_reduce(identity_reduce, mock_gene_embeddings, mock_expression_embeddings):
    gene_embedding_layer = Embedding.from_pretrained(mock_gene_embeddings)
    expression_embedding_layer = Embedding.from_pretrained(mock_expression_embeddings)

    mock_identity_inputs = torch.tensor([0, 0, 2])
    mock_expression_inputs = torch.tensor([0, 0, 2])

    cell_embeddings = identity_reduce(
        mock_identity_inputs,
        gene_embedding_layer,
        mock_expression_inputs,
        expression_embedding_layer,
        None,
    )

    for identity_input, cell_embedding in zip(mock_identity_inputs, cell_embeddings):
        assert torch.allclose(mock_gene_embeddings[identity_input], cell_embedding)
