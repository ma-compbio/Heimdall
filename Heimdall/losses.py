from torch import Tensor, nn


class MaskedLossMixin:
    def __init__(
        self,
        reduction: str = "mean",
        **kwargs,
    ):
        super().__init__(reduction="none", **kwargs)
        self._setup_reduction(reduction)

    def _setup_reduction(self, reduction):
        if reduction == "mean":
            self._reduce = self._reduce_mean
        elif reduction == "sum":
            self._reduce = self._reduce_sum
        else:
            raise ValueError(
                f"Unknown reduction option {reduction!r}. Available options are: 'mean', 'sum'",
            )

    def _reduce_mean(self, loss_mat: Tensor, mask: Tensor) -> Tensor:
        sizes = (~mask).sum(1, keepdim=True)
        return (loss_mat / sizes)[~mask].mean()

    def _reduce_sum(self, loss_mat: Tensor, mask: Tensor) -> Tensor:
        return (loss_mat[~mask] / loss_mat.shape[0]).sum()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = target.isnan()
        target[mask] = 0
        loss_mat = super().forward(input, target)
        loss = self._reduce(loss_mat, mask)
        return loss


class MaskedBCEWithLogitsLoss(MaskedLossMixin, nn.BCEWithLogitsLoss):
    """BCEWithLogitsLoss evaluated on unmasked entires."""
