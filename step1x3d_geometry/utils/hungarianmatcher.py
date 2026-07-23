import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatcherWithLoss(nn.Module):
    """
    This class integrates the Hungarian matcher and loss computation together.
    It processes [batch, tokens1, channel] and [batch, tokens2, channel] features,
    building the cost matrix with cosine similarity and computing the loss with L1 distance
    """

    def __init__(self, cost_cosine: float = 1.0):
        """
        Create the matcher and loss calculator

        Args:
            cost_cosine: Relative weight of cosine similarity in the matching cost
        """
        super().__init__()
        self.matcher = HungarianMatcher(cost_l2=cost_cosine)

    def forward(self, features1, features2):
        """
        Perform matching and compute the batch loss

        Args:
            features1: The first feature sequence, with shape [batch, tokens1, channel]
            features2: The second feature sequence, with shape [batch, tokens2, channel]

        Returns:
            A dict containing:
                - 'loss': A scalar tensor, the average loss over the whole batch
                - 'distances': A list, each element is the L1 distance of the matched pairs in the corresponding batch
                - 'indices': Matching indices, used for subsequent analysis
        """
        # Perform Hungarian matching
        indices = self.matcher(features1, features2)

        #print(indices)

        # Compute the cosine distance of the matched pairs
        distances = self.compute_matched_distance(features1, features2, indices)

        # Compute the loss (average cosine distance)
        loss = self.compute_loss(distances)

        return loss

    def compute_matched_distance(self, features1, features2, indices):
        """Compute the cosine distance of the matched pairs"""
        batch_size = features1.shape[0]
        distances = []

        for b in range(batch_size):
            idx1, idx2 = indices[b]

            # Extract the matched tokens
            feat1_matched = features1[b][idx1]  # [num_matches, channel]
            feat2_matched = features2[b][idx2]  # [num_matches, channel]

            # Compute the L1 distance
            dist = F.cosine_similarity(feat1_matched, feat2_matched,dim=-1).mean(dim=-1)
            distances.append(dist)

        return distances

    def compute_loss(self, distances):
        """
        Compute the loss:
        Use the mean of the L1 distances, since we want similar feature pairs to have smaller distances
        """
        # Gather distances from all batches
        all_dists = torch.stack(distances)

        # Compute the average loss
        if all_dists.numel() == 0:
            dtype = distances[0].dtype if distances else torch.float32
            return torch.tensor(0.0, dtype=dtype, device=distances[0].device if distances else None)

        # Compute the average L1 distance as the loss
        cos_loss = all_dists.mean()
        loss = 1 - cos_loss
        return loss


class HungarianMatcher(nn.Module):
    """Internal Hungarian matcher that builds the cost matrix using L2 distance (Euclidean distance)"""

    def __init__(self, cost_l2: float = 1.0):
        super().__init__()
        self.cost_l2 = cost_l2
        assert cost_l2 > 0, "The weight of the L2 distance cannot be 0"

    @torch.no_grad()
    def forward(self, features1, features2):
        batch_size, tokens1, channels = features1.shape
        tokens2 = features2.shape[1]

        # Ensure the inputs are valid
        assert features1.shape[0] == features2.shape[0], "The batch sizes of the two features must be identical"
        assert features1.shape[2] == features2.shape[2], "The number of channels of the two features must be identical"
        assert tokens1 > 0 and tokens2 > 0, "The feature sequences cannot contain zero tokens"

        # --- Modified section: compute the L2 distance matrix ---
        # Use torch.cdist to compute the Euclidean distance between the two sets of vectors (batch, tokens1, tokens2)
        # p=2 represents the L2 norm
        dist_matrix = torch.cdist(features1, features2, p=2)

        # The cost is directly equal to the L2 distance
        cost_matrix = self.cost_l2 * dist_matrix

        # Perform matching
        indices = []
        cost_matrix_cpu = cost_matrix.cpu()

        for b in range(batch_size):
            row_ind, col_ind = linear_sum_assignment(cost_matrix_cpu[b])
            indices.append(
                (torch.as_tensor(row_ind, dtype=torch.int64, device=features1.device),
                 torch.as_tensor(col_ind, dtype=torch.int64, device=features1.device))
            )

        return indices