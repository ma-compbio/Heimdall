import torch
import torch.nn.functional as F  # noqa: N812
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

    def forward(self, input_: Tensor, target: Tensor) -> Tensor:
        mask = target.isnan()
        target[mask] = 0
        loss_mat = super().forward(input_, target)
        loss = self._reduce(loss_mat, mask)
        return loss


class MaskedBCEWithLogitsLoss(MaskedLossMixin, nn.BCEWithLogitsLoss):
    """BCEWithLogitsLoss evaluated on unmasked entires."""


class CrossEntropyFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        """
        Args:
            alpha (float or list): Balancing factor for each class. If a single float, applies to class 1.
            gamma (float): Modulating factor to down-weight easy samples.
            reduction (str): 'none', 'mean', or 'sum'.
        """
        super().__init__()
        raise NotImplementedError("This class is not implemented correctly yet.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (not probabilities) with shape (batch_size, num_classes)
            targets: Class indices with shape (batch_size,)
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # Standard cross-entropy loss
        p_t = torch.exp(-ce_loss)  # Compute p_t = exp(-CE) (probability of the true class)
        focal_loss = (self.alpha * (1 - p_t) ** self.gamma) * ce_loss  # Apply focal loss formula

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss  # If reduction='none'
