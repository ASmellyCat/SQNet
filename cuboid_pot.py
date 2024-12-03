import os
import warnings
import torch
import numpy as np
import torch.nn as nn
import trimesh
from dgcnn import DGCNNFeat
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from torch.nn.functional import grid_sample
from torch.cuda.amp import autocast, GradScaler

# ===========================
# Suppress Warnings
# ===========================
os.environ['LOKY_MAX_CPU_COUNT'] = '16'  # Replace '16' with the number of physical cores you have
os.environ['OMP_NUM_THREADS'] = '1'      # Or set to the desired number of threads
warnings.filterwarnings('ignore', category=UserWarning)

# ===========================
# Configurable Hyperparameters
# ===========================
num_epochs = 1000
num_cuboids = 6
learning_rate = 0.0001
bsmin_k = 22
coverage_weight = 0.01       # Increased coverage weight
rotation_weight = 0.05
repulsion_weight = 0.05      # Added repulsion weight
consistency_weight = 0.1     # Consistency loss weight
num_surface_points = 1000    # Number of points to sample per cuboid surface
object_names = ["pot"]      # List of object names to process
use_sdf_training = False     # Boolean flag to control SDF network training
output_dir = "./output"
# ===========================

def bsmin(a, dim, k=bsmin_k, keepdim=False): 
    """
    Smooth minimum function for better blending between cuboids.
    Lower k value means smoother transitions between cuboids.
    """
    dmix = -torch.logsumexp(-k * a, dim=dim, keepdim=keepdim) / k
    return dmix

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    Args:
        q: tensor of shape (..., 4), where the last dimension represents (w, x, y, z)
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Normalize the quaternion
    q = q / q.norm(p=2, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    B = q.shape[:-1]

    # Compute rotation matrix elements
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z
    yz = y * z

    rot = torch.stack([
        ww + xx - yy - zz, 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     ww - xx + yy - zz, 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     ww - xx - yy + zz
    ], dim=-1).reshape(*B, 3, 3)

    return rot

def determine_cuboid_sdf(query_points, cuboid_params):
    """
    Compute the SDF between query points and cuboids.

    Args:
        query_points (torch.Tensor): Nx3 tensor of query points.
        cuboid_params (torch.Tensor): Kx10 tensor of cuboid parameters (center, quaternion, dimensions).

    Returns:
        torch.Tensor: Signed distance field of each cuboid primitive with respect to each query point. NxK tensor.
    """
    # Extract cuboid parameters
    cuboid_centers = cuboid_params[:, :3]       # K x 3
    cuboid_quaternions = cuboid_params[:, 3:7]  # K x 4
    cuboid_dimensions = cuboid_params[:, 7:]    # K x 3 (dx, dy, dz)

    # Normalize quaternions
    cuboid_quaternions = cuboid_quaternions / torch.norm(cuboid_quaternions, dim=1, keepdim=True)

    # Compute rotation matrices
    rotation_matrices = quaternion_to_rotation_matrix(cuboid_quaternions)  # K x 3 x 3
    rotation_matrices_inv = rotation_matrices.transpose(1, 2)  # Inverse rotation

    # Expand dimensions for broadcasting
    query_points_expanded = query_points.unsqueeze(1)  # N x 1 x 3
    cuboid_centers_expanded = cuboid_centers.unsqueeze(0)  # 1 x K x 3

    # Translate points to cuboid's local frame
    local_points = query_points_expanded - cuboid_centers_expanded  # N x K x 3

    # Rotate points to align with cuboid's axes
    local_points = torch.einsum('nki,kij->nkj', local_points, rotation_matrices_inv)

    # Compute SDF for axis-aligned box centered at origin
    half_dims = cuboid_dimensions.unsqueeze(0) / 2  # 1 x K x 3
    q = torch.abs(local_points) - half_dims  # N x K x 3

    outside_distance = torch.norm(torch.clamp(q, min=0.0), dim=2)
    inside_distance = torch.clamp(torch.max(q, dim=2)[0], max=0.0)
    sdf = outside_distance + inside_distance  # N x K

    return sdf

class Decoder(nn.Module):
    def __init__(self, num_cuboids=num_cuboids):
        super(Decoder, self).__init__()
        self.num_cuboids = num_cuboids
        in_ch = 256
        feat_ch = 512
        out_ch = num_cuboids * 10  # 10 parameters per cuboid

        self.net1 = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Linear(in_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
            nn.ReLU(inplace=True),
        )

        self.net2 = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.utils.parametrizations.weight_norm(nn.Linear(feat_ch, feat_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(feat_ch, out_ch),
        )
        num_params = sum(p.numel() for p in self.parameters())
        print("[Number of parameters: {}]".format(num_params))

    def forward(self, z):
        in1 = z
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        return out2

class CuboidNet(nn.Module):
    def __init__(self, num_cuboids=num_cuboids):
        super(CuboidNet, self).__init__()
        self.num_cuboids = num_cuboids
        self.encoder = DGCNNFeat(global_feat=True)
        self.decoder = Decoder(num_cuboids=num_cuboids)

    def forward(self, surface_points, query_points):
        features = self.encoder(surface_points)
        cuboid_params = self.decoder(features)

        # Reshape and process cuboid parameters
        cuboid_params = cuboid_params.view(-1, 10)  # num_cuboids x 10

        # Process positions
        cuboid_centers = torch.sigmoid(cuboid_params[:, :3])
        center_adder = torch.tensor([-1.0, -1.0, -1.0], device=cuboid_centers.device)
        center_multiplier = torch.tensor([2.0, 2.0, 2.0], device=cuboid_centers.device)
        cuboid_centers = cuboid_centers * center_multiplier + center_adder

        # Process quaternions
        cuboid_quaternions = cuboid_params[:, 3:7]
        cuboid_quaternions = cuboid_quaternions / torch.norm(cuboid_quaternions, dim=1, keepdim=True)

        # Process dimensions
        cuboid_dimensions = torch.sigmoid(cuboid_params[:, 7:])
        dimension_adder = torch.tensor([0.01, 0.01, 0.01], device=cuboid_dimensions.device)  # Allow smaller dimensions
        dimension_multiplier = torch.tensor([5.0, 5.0, 5.0], device=cuboid_dimensions.device)  # Allow larger dimensions
        cuboid_dimensions = cuboid_dimensions * dimension_multiplier + dimension_adder

        # Combine processed parameters
        cuboid_params = torch.cat([cuboid_centers, cuboid_quaternions, cuboid_dimensions], dim=1)

        # Compute SDF between query points and cuboids
        cuboid_sdf = determine_cuboid_sdf(query_points, cuboid_params)
        return cuboid_sdf, cuboid_params

# def compute_coverage_loss(cuboid_params, surface_points):
#     """
#     Compute coverage loss to encourage cuboids to cover the target surface points.

#     Args:
#         cuboid_params (torch.Tensor): K x 10 tensor of cuboid parameters
#         surface_points (torch.Tensor): N x 3 tensor of surface points

#     Returns:
#         torch.Tensor: Coverage loss
#     """
#     sdf = determine_cuboid_sdf(surface_points, cuboid_params)
#     min_distances = torch.min(sdf, dim=-1)[0]  # Minimum distance to any cuboid
#     coverage_loss = torch.mean(min_distances ** 2)
#     return coverage_loss

def compute_coverage_loss(cuboid_params, surface_points):
    """
    Compute coverage loss following the paper's formulation:
    L1({(zm, qm, tm)}, O) = Ep~S(O)||C(p; ∪Pm)||²

    Args:
        cuboid_params: tensor of cuboid parameters (centers, quaternions, dimensions)
        surface_points: points sampled from the object surface (representing S(O))
    """
    # Compute SDF for surface points
    sdf = determine_cuboid_sdf(surface_points, cuboid_params)
    min_distances = bsmin(sdf, dim=-1)
    
    # Only consider points outside the predicted shape
    min_distances_clamped = torch.clamp(min_distances, min=0.0)
    
    # Compute coverage loss without weighting
    coverage_loss = torch.mean(min_distances_clamped ** 2)
    
    return coverage_loss
def compute_repulsion_loss(cuboid_params):
    """
    Compute repulsion loss to penalize overlapping cuboids, approximated by spheres encompassing the cuboids.

    Args:
        cuboid_params (torch.Tensor): K x 10 tensor of cuboid parameters

    Returns:
        torch.Tensor: Repulsion loss
    """
    centers = cuboid_params[:, :3]  # K x 3
    dimensions = cuboid_params[:, 7:]  # K x 3

    # Compute a radius for each cuboid, e.g., half of the diagonal of the dimensions
    half_diagonals = torch.norm(dimensions / 2, dim=1)  # K

    # Compute pairwise distances between centers
    distances = torch.cdist(centers.unsqueeze(0), centers.unsqueeze(0)).squeeze(0)  # K x K

    # Compute sum of radii for each pair
    radii_sum = half_diagonals.unsqueeze(1) + half_diagonals.unsqueeze(0)  # K x K

    # Compute overlap amount
    overlap = radii_sum - distances  # K x K

    # Exclude self-overlaps by zeroing the diagonal
    overlap.fill_diagonal_(0)

    # Mask to consider only the upper triangle (since the matrix is symmetric)
    mask = torch.triu(torch.ones_like(overlap), diagonal=1).bool()

    # Only consider positive overlaps (where overlap > 0)
    overlap = torch.clamp(overlap[mask], min=0)

    # Repulsion loss is proportional to the square of the overlap amount
    repulsion_loss = torch.mean(overlap ** 2)

    return repulsion_loss

def sample_points_on_cuboid_surface(dimensions, num_points):
    """
    Sample points on the surface of a cuboid with given dimensions.

    Args:
        dimensions (torch.Tensor): Tensor of shape (3,) representing (dx, dy, dz)
        num_points (int): Number of points to sample

    Returns:
        torch.Tensor: Tensor of shape (num_points, 3) representing points on the surface
    """
    # dimensions: (3,)
    device = dimensions.device

    # We have 6 faces: +/-x, +/-y, +/-z

    # First, compute areas of each face to sample proportionally
    dx, dy, dz = dimensions / 2  # half dimensions

    # Areas of faces
    area_xy = dx * dy * 2  # faces perpendicular to z
    area_xz = dx * dz * 2  # faces perpendicular to y
    area_yz = dy * dz * 2  # faces perpendicular to x

    total_area = area_xy + area_xz + area_yz

    # Number of points per face
    num_points_xy = int(num_points * area_xy / total_area)
    num_points_xz = int(num_points * area_xz / total_area)
    num_points_yz = num_points - num_points_xy - num_points_xz  # remaining points

    # For each face, sample points uniformly

    # Face +/-z (x and y vary, z is constant)
    # Sample num_points_xy // 2 points on z = +1
    u1 = torch.rand(num_points_xy // 2, 1, device=device) * 2 - 1  # [-1, 1]
    v1 = torch.rand(num_points_xy // 2, 1, device=device) * 2 - 1
    z1 = torch.ones(num_points_xy // 2, 1, device=device)  # z = +1
    points1 = torch.cat([u1, v1, z1], dim=1)

    # Sample num_points_xy - num_points_xy // 2 points on z = -1
    u2 = torch.rand(num_points_xy - num_points_xy // 2, 1, device=device) * 2 - 1  # [-1, 1]
    v2 = torch.rand(num_points_xy - num_points_xy // 2, 1, device=device) * 2 - 1
    z2 = -torch.ones(num_points_xy - num_points_xy // 2, 1, device=device)  # z = -1
    points2 = torch.cat([u2, v2, z2], dim=1)

    # Faces x = +1 and x = -1
    v3 = torch.rand(num_points_yz // 2, 1, device=device) * 2 - 1  # [-1, 1]
    w3 = torch.rand(num_points_yz // 2, 1, device=device) * 2 - 1
    x3 = torch.ones(num_points_yz // 2, 1, device=device)  # x = +1
    points3 = torch.cat([x3, v3, w3], dim=1)

    v4 = torch.rand(num_points_yz - num_points_yz // 2, 1, device=device) * 2 - 1
    w4 = torch.rand(num_points_yz - num_points_yz // 2, 1, device=device) * 2 - 1
    x4 = -torch.ones(num_points_yz - num_points_yz // 2, 1, device=device)
    points4 = torch.cat([x4, v4, w4], dim=1)

    # Faces y = +1 and y = -1
    u5 = torch.rand(num_points_xz // 2, 1, device=device) * 2 - 1
    w5 = torch.rand(num_points_xz // 2, 1, device=device) * 2 - 1
    y5 = torch.ones(num_points_xz // 2, 1, device=device)  # y = +1
    points5 = torch.cat([u5, y5, w5], dim=1)

    u6 = torch.rand(num_points_xz - num_points_xz // 2, 1, device=device) * 2 - 1
    w6 = torch.rand(num_points_xz - num_points_xz // 2, 1, device=device) * 2 - 1
    y6 = -torch.ones(num_points_xz - num_points_xz // 2, 1, device=device)
    points6 = torch.cat([u6, y6, w6], dim=1)

    # Concatenate all points
    points = torch.cat([points1, points2, points3, points4, points5, points6], dim=0)  # (num_points, 3)

    # Now scale points by dimensions / 2
    points_scaled = points * (dimensions / 2).unsqueeze(0)  # (num_points, 3)

    return points_scaled

def compute_consistency_loss(cuboid_params, sdf_function, num_surface_points_per_cuboid):
    """
    Compute the consistency loss between the cuboids and the target object's SDF.

    Args:
        cuboid_params (torch.Tensor): K x 10 tensor of cuboid parameters
        sdf_function (callable): Function that takes points (N x 3) and returns SDF values (N)
        num_surface_points_per_cuboid (int): Number of surface points to sample per cuboid

    Returns:
        torch.Tensor: Consistency loss
    """
    K = cuboid_params.shape[0]
    losses = []
    for m in range(K):
        center = cuboid_params[m, :3]  # (3,)
        quaternion = cuboid_params[m, 3:7]  # (4,)
        dimensions = cuboid_params[m, 7:]  # (3,)

        # Sample points p' on the surface of P_m
        p_prime = sample_points_on_cuboid_surface(dimensions, num_surface_points_per_cuboid).to(cuboid_params.device)  # (N_p, 3)

        # Rotate p' using quaternion
        rotation_matrix = quaternion_to_rotation_matrix(quaternion.unsqueeze(0))  # (1, 3, 3)
        p_rotated = torch.matmul(p_prime, rotation_matrix.squeeze(0).T)  # (N_p, 3)

        # Translate
        p = p_rotated + center.unsqueeze(0)  # (N_p, 3)

        # Evaluate SDF at p
        C_p = sdf_function(p)  # (N_p,)

        # Compute loss as mean squared value of C_p
        loss_m = torch.mean(C_p ** 2)

        losses.append(loss_m)

    consistency_loss = torch.sum(torch.stack(losses))

    return consistency_loss

def visualise_cuboids(cuboid_params, reference_model, save_path=None):
    """
    Visualize the cuboids with distinct colors and optionally save the visualization.

    Args:
        cuboid_params (torch.Tensor): Kx10 tensor of cuboid parameters.
        reference_model (trimesh.Trimesh): Reference mesh model to display alongside cuboids.
        save_path (str, optional): Path to save the visualized cuboids.
    """
    cuboid_params = cuboid_params.cpu().detach().numpy()
    cuboid_centers = cuboid_params[..., :3]
    cuboid_quaternions = cuboid_params[..., 3:7]
    cuboid_dimensions = cuboid_params[..., 7:]

    # Define a list of distinct colors (RGBA)
    colors = [
        [179, 0, 30, 255],    # Red
        [179, 255, 25, 255],  # Green
        [255, 224, 58, 255],  # Yellow
        [128, 170, 255, 255], # Cyan
        [255, 102, 255, 255], # Magenta
        [255, 119, 51, 255],  # Orange
        [196, 77, 255, 255],  # Purple
        [179, 242, 255, 255], # Teal
        [128, 128, 0, 255],   # Olive
    ]

    scene = trimesh.Scene()
    if reference_model is not None:
        if isinstance(reference_model, trimesh.points.PointCloud):
            # Assign a color to the reference point cloud
            reference_model.colors = [[219, 204, 188, 255]] * len(reference_model.vertices)
        scene.add_geometry(reference_model)

    for i, (center, quaternion, dimensions) in enumerate(zip(cuboid_centers, cuboid_quaternions, cuboid_dimensions)):
        box = trimesh.creation.box(extents=dimensions)
        transform = trimesh.transformations.quaternion_matrix(quaternion)
        transform[:3, 3] = center
        box.apply_transform(transform)

        # Assign color from the list, cycling if necessary
        color = colors[i % len(colors)]
        box.visual.face_colors = color

        scene.add_geometry(box)

    if save_path is not None:
        scene.export(save_path)
    scene.show()

def initialize_cuboid_params_with_spectral_clustering(model, surface_pointcloud, num_cuboids, device):
    """
    Initialize the decoder's last layer biases using Spectral Clustering and PCA for orientations.

    Args:
        model (nn.Module): The CuboidNet model instance.
        surface_pointcloud (torch.Tensor): Tensor of surface points (N x 3).
        num_cuboids (int): Number of cuboids.
        device (torch.device): The device to run computations on.
    """
    # Number of neighbors for nearest neighbors graph
    k = 30  # Adjust as needed

    # Perform Spectral Clustering to get initial cuboid centers
    clustering = SpectralClustering(
        n_clusters=num_cuboids,
        affinity='nearest_neighbors',
        n_neighbors=k,
        assign_labels='kmeans'
    )
    labels = clustering.fit_predict(surface_pointcloud.cpu().numpy())

    initial_centers = []
    initial_quaternions = []
    for i in range(num_cuboids):
        cluster_points = surface_pointcloud[labels == i]
        if len(cluster_points) > 0:
            # Compute initial center as the mean of each cluster
            center = cluster_points.mean(dim=0)
            initial_centers.append(center)

            # Perform PCA to get the principal axes
            pca = PCA(n_components=3)
            pca.fit(cluster_points.cpu().numpy())
            rotation_matrix = pca.components_.T  # 3 x 3

            # Ensure a right-handed coordinate system
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix[:, -1] *= -1

            # Convert rotation matrix to quaternion
            quaternion = trimesh.transformations.quaternion_from_matrix(rotation_matrix)
            quaternion = torch.from_numpy(quaternion).to(device)
            initial_quaternions.append(quaternion)
        else:
            # If a cluster has no points, initialize randomly
            center = torch.rand(3).to(device) * 2 - 1  # Random in [-1, 1]
            initial_centers.append(center)
            quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # Identity quaternion
            initial_quaternions.append(quaternion)

    initial_centers = torch.stack(initial_centers).to(device)
    initial_quaternions = torch.stack(initial_quaternions).to(device)

    # Compute pre-activation values for centers
    y_centers = (initial_centers + 1.0) / 2.0
    epsilon = 1e-6
    y_centers = torch.clamp(y_centers, epsilon, 1 - epsilon)
    x_centers = torch.log(y_centers / (1 - y_centers))

    # Initial dimensions (you can adjust desired dimensions as needed)
    desired_dimensions = torch.tensor([0.5, 0.5, 0.5], device=device).unsqueeze(0).repeat(num_cuboids, 1)
    dimension_adder = torch.tensor([0.01, 0.01, 0.01], device=device)
    dimension_multiplier = torch.tensor([5.0, 5.0, 5.0], device=device)
    y_dimensions = (desired_dimensions - dimension_adder) / dimension_multiplier
    y_dimensions = torch.clamp(y_dimensions, epsilon, 1 - epsilon)
    x_dimensions = torch.log(y_dimensions / (1 - y_dimensions))

    # Combine pre-activation parameters
    initial_cuboid_params_pre_activation = torch.cat([x_centers, initial_quaternions, x_dimensions], dim=1)  # num_cuboids x 10

    # Flatten to (num_cuboids * 10)
    initial_cuboid_params_pre_activation = initial_cuboid_params_pre_activation.flatten()

    # Set the decoder's last layer bias
    with torch.no_grad():
        model.decoder.net2[-1].bias.copy_(initial_cuboid_params_pre_activation)

class SDFNetwork(nn.Module):
    """
    Neural network to approximate the Signed Distance Function (SDF) of the target object.
    """
    def __init__(self, hidden_size=256, num_layers=4):
        super(SDFNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(3, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

def train_sdf_network(sdf_net, sdf_loader, sdf_optimizer, sdf_num_epochs, device):
    """
    Train the SDF network.

    Args:
        sdf_net (nn.Module): The SDF network instance.
        sdf_loader (DataLoader): DataLoader for SDF training data.
        sdf_optimizer (Optimizer): Optimizer for the SDF network.
        sdf_num_epochs (int): Number of training epochs.
        device (torch.device): Device to train on.
    """
    scaler = torch.amp.GradScaler(device='cuda')
    sdf_net.train()
    for epoch in range(sdf_num_epochs):
        total_loss = 0.0
        for batch_points, batch_values in sdf_loader:
            sdf_optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                pred_values = sdf_net(batch_points)
                loss = torch.mean((pred_values - batch_values) ** 2)
            scaler.scale(loss).backward()
            scaler.step(sdf_optimizer)
            scaler.update()
            total_loss += loss.item() * batch_points.shape[0]
        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_loss = total_loss / len(sdf_loader.dataset)
            print(f"SDF Epoch {epoch + 1}/{sdf_num_epochs}, Loss: {avg_loss:.6f}")
    # Save the trained SDF network
    os.makedirs(os.path.join(output_dir, "model", "sdfnet"), exist_ok=True)
    torch.save(sdf_net.state_dict(), os.path.join(output_dir, "model", "sdfnet", "sdf_network.pth"))

def load_trained_sdf_network(sdf_net, path, device):
    """
    Load a pre-trained SDF network.

    Args:
        sdf_net (nn.Module): The SDF network instance.
        path (str): Path to the saved SDF network state dict.
        device (torch.device): Device to load the network on.

    Returns:
        nn.Module: Loaded SDF network.
    """
    sdf_net.load_state_dict(torch.load(path, map_location=device))
    sdf_net.eval()
    return sdf_net

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    dataset_root_path = "./reference_models_processed"  # Root directory for the dataset

    for obj_name in object_names:
        print(f"Processing object: {obj_name}")
        
        object_path = os.path.join(dataset_root_path, obj_name)
        mesh_model_path = os.path.join(object_path, f"{obj_name}.obj")
        pcd_model_path = os.path.join(object_path, "surface_points.ply")
        sdf_npz_path = os.path.join(object_path, "voxel_and_sdf.npz")
        
        # Load mesh model
        mesh_model = trimesh.load(mesh_model_path)
        if isinstance(mesh_model, list):
            print(f"{obj_name}.obj contains multiple meshes. Selecting the first one.")
            mesh_model = mesh_model[0]  # Select the first mesh
        
        # Load point cloud model
        pcd_model = trimesh.load(pcd_model_path)
        if isinstance(pcd_model, list):
            print(f"surface_points.ply for {obj_name} contains multiple point clouds. Selecting the first one.")
            pcd_model = pcd_model[0]  # Select the first point cloud
        
        # Load SDF data
        data_npz = np.load(sdf_npz_path)
        points, values = data_npz["sdf_points"], data_npz["sdf_values"]
        points = torch.from_numpy(points).float().to(device)
        values = torch.from_numpy(values).float().to(device)
        surface_pointcloud = torch.from_numpy(pcd_model.vertices).float().to(device)

        # Initialize CuboidNet
        model = CuboidNet(num_cuboids).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # ================================
        # Initialize SDF Network
        # ================================
        sdf_net = SDFNetwork().to(device)
        sdf_model_dir = os.path.join(output_dir, "model", "sdfnet")
        os.makedirs(sdf_model_dir, exist_ok=True)
        sdf_model_path = os.path.join(sdf_model_dir, "sdf_network.pth")

        if use_sdf_training:
            if os.path.exists(sdf_model_path):
                print("Loading pre-trained SDF network...")
                sdf_net = load_trained_sdf_network(sdf_net, sdf_model_path, device)
            else:
                print("Training SDF network...")
                sdf_optimizer = torch.optim.Adam(sdf_net.parameters(), lr=1e-3)
                sdf_num_epochs = 1000
                sdf_batch_size = 1024

                sdf_dataset = torch.utils.data.TensorDataset(points, values)
                sdf_loader = torch.utils.data.DataLoader(sdf_dataset, batch_size=sdf_batch_size, shuffle=True)

                train_sdf_network(sdf_net, sdf_loader, sdf_optimizer, sdf_num_epochs, device)
        else:
            if os.path.exists(sdf_model_path):
                print("Loading pre-trained SDF network...")
                sdf_net = load_trained_sdf_network(sdf_net, sdf_model_path, device)
            else:
                raise FileNotFoundError(f"SDF network not found at {sdf_model_path}. Please set use_sdf_training=True to train it.")

        # Define sdf_function
        def sdf_function(p):
            with torch.no_grad():
                return sdf_net(p)

        # ================================
        # Spectral Clustering Initialization
        # ================================
        initialize_cuboid_params_with_spectral_clustering(model, surface_pointcloud, num_cuboids, device)
        # ================================

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            cuboid_sdf, cuboid_params = model(
                surface_pointcloud.unsqueeze(0).transpose(2, 1), points
            )

            # Main SDF loss
            cuboid_sdf = bsmin(cuboid_sdf, dim=-1, k=bsmin_k).to(device)  
            mse_loss = torch.mean((cuboid_sdf - values) ** 2)

            # Quaternion regularization
            quaternions = cuboid_params[:, 3:7]
            rotation_loss = torch.mean(torch.abs(quaternions[:, 1:]))

            # Coverage loss
            coverage_loss = compute_coverage_loss(cuboid_params, surface_pointcloud)

            # Repulsion loss
            repulsion_loss = compute_repulsion_loss(cuboid_params)

            # Consistency loss
            consistency_loss = compute_consistency_loss(cuboid_params, sdf_function, num_surface_points)

            # Combined loss with weights
            loss = mse_loss + \
                rotation_weight * rotation_loss + \
                coverage_weight * coverage_loss + \
                repulsion_weight * repulsion_loss + \
                consistency_weight * consistency_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                dimensions = cuboid_params[:, 7:]  # K x 3
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Total Loss: {loss.item():.6f}, "
                      f"MSE Loss: {mse_loss.item():.6f}, "
                      f"Rotation Loss: {rotation_loss.item():.6f}, "
                      f"Coverage Loss: {coverage_loss.item():.6f}, "
                      f"Repulsion Loss: {repulsion_loss.item():.6f}, "
                      f"Consistency Loss: {consistency_loss.item():.6f}")

        # Save the cuboid parameters
        os.makedirs(output_dir, exist_ok=True)
        cuboid_params_path = os.path.join(output_dir, obj_name, f"{obj_name}_cuboid_params.npy")
        np.save(cuboid_params_path, cuboid_params.cpu().detach().numpy())

        print("Final Cuboid Parameters:")
        print(cuboid_params)

        # Visualize the cuboids with distinct colors
        cuboids_save_path = os.path.join(output_dir, obj_name, f"{obj_name}_cuboids.obj")
        visualise_cuboids(cuboid_params, reference_model=pcd_model, save_path=cuboids_save_path)

    print("Processing Completed.")

if __name__ == "__main__":
    main()
