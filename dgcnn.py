# dgcnn.py

import torch
import torch.nn as nn


def knn(x, k):
    """
    Compute the k-nearest neighbors for each point in the point cloud.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_dims, num_points).
        k (int): Number of nearest neighbors to find.

    Returns:
        torch.Tensor: Indices of the k-nearest neighbors for each point. Shape: (batch_size, num_points, k)
    """
    # Compute pairwise squared Euclidean distances
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # Shape: (batch_size, num_points, num_points)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)     # Shape: (batch_size, 1, num_points)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # Shape: (batch_size, num_points, num_points)

    # Retrieve the indices of the k-smallest distances (nearest neighbors)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # Shape: (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    """
    Generate graph features for each point in the point cloud by concatenating the differences with its k-nearest neighbors.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_dims, num_points).
        k (int, optional): Number of nearest neighbors to consider. Defaults to 20.
        idx (torch.Tensor, optional): Precomputed indices of nearest neighbors. If None, computed using knn. Defaults to None.
        dim9 (bool, optional): If True, computes knn based on the last 3 dimensions. Defaults to False.

    Returns:
        torch.Tensor: Graph features of shape (batch_size, 2 * num_dims, num_points, k).
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # Ensure x is of shape (batch_size, num_dims, num_points)

    if idx is None:
        if not dim9:
            idx = knn(x, k=k)   # Compute k-nearest neighbors based on all dimensions
        else:
            idx = knn(x[:, 6:], k=k)  # Compute k-nearest neighbors based on the last 3 dimensions

    device = x.device
    # Compute base indices for batch-wise indexing
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base  # Adjust indices for each batch
    idx = idx.view(-1)     # Flatten for indexing

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # Shape: (batch_size, num_points, num_dims)
    # Gather the neighbor points based on the indices
    feature = x.view(batch_size * num_points, -1)[idx, :]  # Shape: (batch_size * num_points * k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # Shape: (batch_size, num_points, k, num_dims)

    # Repeat the original points to concatenate with neighbor differences
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # Shape: (batch_size, num_points, k, num_dims)
    # Concatenate the differences with the original points
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # Shape: (batch_size, 2 * num_dims, num_points, k)

    return feature  # Shape: (batch_size, 2 * num_dims, num_points, k)


class DGCNNFeat(nn.Module):
    """
    Dynamic Graph CNN Feature Extractor.

    This module extracts features from point clouds using dynamic graph-based convolutional layers.
    It is inspired by the DGCNN architecture and is used as the encoder in the SQNet model.

    Args:
        k (int, optional): Number of nearest neighbors to consider. Defaults to 20.
        emb_dims (int, optional): Dimension of the embedding/features. Defaults to 256.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        global_feat (bool, optional): If True, returns global features by taking the maximum across points. If False, returns per-point features. Defaults to True.
    """
    def __init__(self, k=20, emb_dims=256, dropout=0.5, global_feat=True):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.dropout = dropout
        self.global_feat = global_feat

        # Define BatchNorm layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        # Define convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),  # Input channels: 2 * num_dims (num_dims=3)
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),  # Concatenated features
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),  # Concatenated features
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),  # Concatenated global features
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        """
        Forward pass of the DGCNN feature extractor.

        Args:
            x (torch.Tensor): Input point cloud tensor of shape (batch_size, num_dims, num_points).

        Returns:
            torch.Tensor: Extracted features. If global_feat is True, shape is (batch_size, emb_dims).
                          If global_feat is False, shape is (batch_size, emb_dims, num_points).
        """
        batch_size = x.size(0)
        num_points = x.size(2)

        # Generate graph features by computing differences with k-nearest neighbors
        x = get_graph_feature(x, k=self.k, dim9=False)   # Shape: (batch_size, 6, num_points, k)
        x = self.conv1(x)                                # Shape: (batch_size, 64, num_points, k)
        x = self.conv2(x)                                # Shape: (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]             # Shape: (batch_size, 64, num_points)

        # Repeat the process on the new features
        x = get_graph_feature(x1, k=self.k)              # Shape: (batch_size, 128, num_points, k)
        x = self.conv3(x)                                # Shape: (batch_size, 64, num_points, k)
        x = self.conv4(x)                                # Shape: (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]             # Shape: (batch_size, 64, num_points)

        # Repeat the process once more
        x = get_graph_feature(x2, k=self.k)              # Shape: (batch_size, 128, num_points, k)
        x = self.conv5(x)                                # Shape: (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]             # Shape: (batch_size, 64, num_points)

        # Concatenate all intermediate features
        x = torch.cat((x1, x2, x3), dim=1)               # Shape: (batch_size, 192, num_points)

        # Final convolution to obtain the embedding
        x = self.conv6(x)                                # Shape: (batch_size, emb_dims, num_points)
        if self.global_feat:
            x = x.max(dim=-1)[0]                          # Shape: (batch_size, emb_dims)
        return x                                         # Shape: (batch_size, emb_dims) if global_feat else (batch_size, emb_dims, num_points)
